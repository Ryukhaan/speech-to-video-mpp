import glob
from os.path import dirname, join, basename, isfile

# Training Tensorboard
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import json
import gc
import torch
from torch import nn
from torch import optim
import torchvision
from torchsummary import summary
from torch.utils import data as data_utils
from librosa import get_duration
import clip
import numpy as np
import cv2, os, sys, subprocess, platform, torch
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat
from scipy.io import wavfile

from pydub import AudioSegment
from pydub.utils import make_chunks

import matplotlib.pyplot as plt
from scipy.signal import correlate

# Audio framework and Encodec
#from encodec import EncodecModel
#from encodec.utils import convert_audio
import torchaudio

from sklearn.model_selection import train_test_split

from models.LNet import LNet
import pickle
from models import losses
import preprocessing.facing as preprocessing

sys.path.append('third_part')
# 3dmm extraction
from third_part.face3d.util.preprocess import align_img
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.extract_kp_videos import KeypointExtractor
# face enhancement
#from third_part.GPEN.face_enhancement import FaceEnhancement
#from third_part.GFPGAN.gfpgan import GFPGANer
# expression control
from third_part.ganimation_replicate.model.ganimation import GANimationModel

from futils import audio
from futils.ffhq_preprocess import Croper
from futils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from futils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_train_model, \
    split_coeff, \
    trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict, save_checkpoint, load_model
from futils.inference_utils import load_model as fu_load_model
from futils import hparams, audio
import warnings
import argparse
warnings.filterwarnings("ignore")


def options():
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--DNet_path', type=str, default='checkpoints/DNet.pt')
    parser.add_argument('--face3d_net_path', type=str, default='checkpoints/face3d_pretrain_epoch_20.pth')
    parser.add_argument('--exp_img', type=str, help='Expression template. neutral, smile or image path', default=None)
    parser.add_argument('--outfile', type=str, help='Video path to save result')
    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                        default=25., required=False)
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 20, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')
    parser.add_argument('--face_det_batch_size', type=int, help='Batch size for face detection', default=4)
    parser.add_argument('--LNet_batch_size', type=int, help='Batch size for LNet', default=16)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                             'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                             'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')
    parser.add_argument('--static', default=False, action='store_true')
    parser.add_argument('--up_face', default='original')
    parser.add_argument('--one_shot', action='store_true')
    parser.add_argument('--without_rl1', default=False, action='store_true', help='Do not use the relative l1')
    parser.add_argument('--tmp_dir', type=str, default='temp', help='Folder to save tmp results')
    parser.add_argument('--re_preprocess', action='store_true')
    parser.add_argument('--cropped_image', default=False, action='store_true',
                        help='Cropped the mouth to paste on the original video')
    parser.add_argument('--dict_path', default="", help="Path to phones dictionary")
    parser.add_argument('--json_path', default="", help="Path to JSON MFA result")
    parser.add_argument('--sync_path', default="checkpoints/lipsync_expert.pth",
                        help="Path to LipSync Network checkpoints")
    args = parser.parse_args()
    return args

args = options()
hparams = hparams.hparams

lnet_T = 5

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Weird function
def get_frame_id(self, frame):
    return int(basename(frame).split('.')[0])

def read_video(dataset, index, args):
    video_stream = cv2.VideoCapture(dataset[index])
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    if os.path.isfile(dataset[index].split('.')[0] + "_cropped.npy"):
        full_frames = np.load(dataset[index].split('.')[0] + "_cropped.npy", allow_pickle=True)
    else:
        full_frames = []
        while True:
           still_reading, frame = video_stream.read()
           if not still_reading:
               video_stream.release()
               break
           y1, y2, x1, x2 = args.crop
           if x2 == -1: x2 = frame.shape[1]
           if y2 == -1: y2 = frame.shape[0]
           frame = frame[y1:y2, x1:x2]
           full_frames.append(frame)
    return full_frames, fps

def landmarks_estimate(dataset, idx, nframes, reprocess=False):
    # face detection & cropping, cropping the first frame as the style of FFHQ
    croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')

    #cv2.imwrite('/home/dremi/check.png', nframes[0])
    #full_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in nframes]
    full_frames_RGB = [np.asarray(frame) for frame in nframes]

    # Why there was a try ?
    full_frames_RGB, crop, quad = croper.crop(full_frames_RGB, xsize=512)  # Why 512 ?

    clx, cly, crx, cry = crop
    lx, ly, rx, ry = quad
    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
    oy1, oy2, ox1, ox2 = cly + ly, min(cly + ry, full_frames_RGB[0].shape[0]), clx + lx, min(clx + rx, full_frames_RGB[0].shape[1])
    coordinates = oy1, oy2, ox1, ox2

    # original_size = (ox2 - ox1, oy2 - oy1)
    frames_pil = [Image.fromarray(cv2.resize(frame, (256, 256))) for frame in full_frames_RGB]

    # get the landmark according to the detected face.
    if not os.path.isfile(dataset[idx].split('.')[0] + '_landmarks.txt') or reprocess:
        torch.cuda.empty_cache()
        kp_extractor = KeypointExtractor()
        lm = kp_extractor.extract_keypoint(frames_pil, dataset[idx].split('.')[0] + '_landmarks.txt')
        del kp_extractor
    else:
        lm = np.loadtxt(dataset[idx].split('.')[0] + '_landmarks.txt').astype(np.float32)
        #lm = lm.reshape([lnet_T, -1, 2])
    return lm, coordinates, frames_pil

def face_3dmm_extraction(dataset, idx, args, frames_pil, lm, reprocess=False):
    torch.cuda.empty_cache()
    if not os.path.isfile(dataset[idx].split('.')[0] + "_coeffs.npy") or reprocess:
        net_recon = load_face3d_net(args.face3d_net_path, device)
        lm3d_std = load_lm3d('checkpoints/BFM')
        video_coeffs = []
        for idx in range(len(frames_pil)):  # , desc="[Step 2] 3DMM Extraction In Video:"):
            frame = frames_pil[idx]
            W, H = frame.size
            lm_idx = lm[idx].reshape([-1, 2])
            if np.mean(lm_idx) == -1:
                lm_idx = (lm3d_std[:, :2] + 1) / 2.
                lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
            else:
                lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

            trans_params, im_idx, lm_idx, _ = align_img(frame, lm_idx, lm3d_std)
            trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
            im_idx_tensor = torch.tensor(np.array(im_idx) / 255., dtype=torch.float32).permute(2, 0, 1).to \
                (device).unsqueeze(0)
            with torch.no_grad():
                coeffs = split_coeff(net_recon(im_idx_tensor))

            pred_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}
            pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'], \
                                         pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
            video_coeffs.append(pred_coeff)
        semantic_npy = np.array(video_coeffs)[:, 0]
        if reprocess:
            np.save(dataset[idx].split('.')[0] + '_coeffs.npy', semantic_npy)
    else:
        semantic_npy = np.load(dataset[idx].split('.')[0] + "_coeffs.npy").astype(np.float32)

    return semantic_npy


def hack_3dmm_expression(dataset, idx, frames_pil, semantic_npy, args, reprocess=False):
    expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_center'])[0]
    D_Net, _ = load_model(args, device)
    # Video Image Stabilized
    if not os.path.isfile(dataset[idx].split('.')[0] + '_stablized.npy'):
        imgs = []
        for idx in range(len(frames_pil)):  # desc="[Step 3] Stablize the expression In Video:"):
            if args.one_shot:
                source_img = trans_image(frames_pil[0]).unsqueeze(0).to(device)
                semantic_source_numpy = semantic_npy[0:1]
            else:
                source_img = trans_image(frames_pil[idx]).unsqueeze(0).to(device)
                semantic_source_numpy = semantic_npy[idx:idx + 1]
            ratio = find_crop_norm_ratio(semantic_source_numpy, semantic_npy)
            coeff = transform_semantic(semantic_npy, idx, ratio).unsqueeze(0).to(device)

            # hacking the new expression
            coeff[:, :64, :] = expression[None, :64, None].to(device)
            with torch.no_grad():
                output = D_Net(source_img, coeff)
            img_stablized = np.uint8 \
                ((output['fake_image'].squeeze(0).permute(1, 2, 0).cpu().clamp_(-1, 1).numpy() + 1) / 2. * 255)
            imgs.append(cv2.cvtColor(img_stablized, cv2.COLOR_RGB2BGR))
        if reprocess:
            np.save(dataset[idx].split('.')[0] + '_stablized.npy', imgs)
        del D_Net
        torch.cuda.empty_cache()
    else:
        imgs = np.load(dataset[idx].split('.')[0] + "_stablized.npy")

    return imgs

def get_full_mels(dataset, idx, imgs, full_frames, lm, fps):
    vidname = dataset[idx]
    wavpath = vidname.split('.')[0] + '.wav'
    wav = audio.load_wav(wavpath, hparams.sample_rate)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_step_size, mel_idx_multiplier, i, mel_chunks = 16, 80. / fps, 0, []
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1
    imgs = imgs[:len(mel_chunks)]
    full_frames = full_frames[:len(mel_chunks)]
    lm = lm[:len(mel_chunks)]
    mel = mel.T

    return imgs, full_frames, lm, mel_chunks

def get_enhanced_imgs(imgs):
    #ref_enhancer = FaceEnhancement(args, base_dir='checkpoints',
    #                               in_size=512, channel_multiplier=2, narrow=1, sr_scale=4,
    #                               model='GPEN-BFR-512', use_sr=False)
    imgs_enhanced = []
    for idx in range(len(imgs)):
        pred = cv2.resize(imgs[idx], (256, 256))
        imgs_enhanced.append(pred)
    return imgs_enhanced

def datagen(frames, mels, full_frames, frames_pil, cox):
    img_batch, mel_batch, frame_batch, coords_batch, ref_batch, full_frame_batch = [], [], [], [], [], []
    base_name = args.face.split('/')[-1]
    refs = []
    image_size = 256

    # original frames
    kp_extractor = KeypointExtractor()
    fr_pil = [Image.fromarray(frame) for frame in frames]
    lms = kp_extractor.extract_keypoint(fr_pil, 'temp/'+base_name+'x12_landmarks.txt')
    frames_pil = [ (lm, frame) for frame,lm in zip(fr_pil, lms)] # frames is the croped version of modified face
    crops, orig_images, quads  = crop_faces(image_size, frames_pil, scale=1.0, use_fa=True)
    inverse_transforms = [calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]) for quad in quads]
    del kp_extractor.detector

    oy1,oy2,ox1,ox2 = cox
    face_det_results = face_detect(full_frames, args, jaw_correction=True)

    for inverse_transform, crop, full_frame, face_det in zip(inverse_transforms, crops, full_frames, face_det_results):
        imc_pil = paste_image(inverse_transform, crop, Image.fromarray(
            cv2.resize(full_frame[int(oy1):int(oy2), int(ox1):int(ox2)], (256, 256))))

        ff = full_frame.copy()
        ff[int(oy1):int(oy2), int(ox1):int(ox2)] = cv2.resize(np.array(imc_pil.convert('RGB')), (ox2 - ox1, oy2 - oy1))
        oface, coords = face_det
        y1, y2, x1, x2 = coords
        refs.append(ff[y1: y2, x1:x2])

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face = refs[idx]
        oface, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))
        oface = cv2.resize(oface, (args.img_size, args.img_size))

        img_batch.append(oface)
        ref_batch.append(face)
        mel_batch.append(m)
        coords_batch.append(coords)
        frame_batch.append(frame_to_save)
        full_frame_batch.append(full_frames[idx].copy())

        if len(img_batch) >= args.LNet_batch_size:
            img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
            img_masked = img_batch.copy()
            img_original = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0
            img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            return img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch
            #img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch, ref_batch  = [], [], [], [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
        img_masked = img_batch.copy()
        img_original = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0
        img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        return img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch

def preprocess(dataset, args):
    # if not os.path.isfile(self.all_videos[self.idx].split('.')[0] +'_cropped.npy'):
    for idx in tqdm(range(len(dataset))):
        # Read the full frame from the video idx
        full_frames, fps = read_video(dataset, idx, args)

        # Get landmarks with face detection (already save
        lm, coordinates, frames_pil = landmarks_estimate(dataset, idx, full_frames, reprocess=args.re_preprocess)

        # Get 3DMM features
        semantic_npy = face_3dmm_extraction(dataset, idx, args, frames_pil, lm, reprocess=args.re_preprocess)

        # Hack the 3DMM features to have neutral face
        imgs = hack_3dmm_expression(dataset, idx, frames_pil, semantic_npy, args, reprocess=args.re_preprocess)

        # Split mel-spectrogram into chunks
        imgs, full_frames, lm, mel_chunks = get_full_mels(dataset, idx, imgs, full_frames, lm, fps)

        # Enhanced images
        imgs_enhanced = get_enhanced_imgs(imgs)

        # Recrop face according to mel chunks
        gen = datagen(imgs_enhanced, mel_chunks, full_frames, None, coordinates)
        img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch = gen
        # Save Images Batch
        if not os.path.isfile(dataset[idx].split('.')[0] + '_img_batch.npy'):
            np.save(dataset[idx].split('.')[0] + '_img_batch.npy', img_batch)
        # Save Mel-Spec Batch
        if not os.path.isfile(dataset[idx].split('.')[0] + '_mel_batch.npy'):
            np.save(dataset[idx].split('.')[0] + '_mel_batch.npy', mel_batch)
        # Save Images Orig Batch
        if not os.path.isfile(dataset[idx].split('.')[0] + '_img_orig.npy'):
            np.save(dataset[idx].split('.')[0] + '_img_orig.npy', img_original)

def get_image_list(data_root, split):
    filelist = []
    with open('./filelists/{}.txt'.format(split)) as f:
        for line in f:
            line = line.rstrip()
            if line.split('.')[-1] == 'mp4':
                filelist.append(os.path.join(data_root, line))
    return filelist

def main():
    # Get all file's name using filelists.txt
    filenames = get_image_list(args.data_root, 'train')

    # Preprocess all files
    preprocess(filenames, args)

if __name__ == "__main__":
    main()