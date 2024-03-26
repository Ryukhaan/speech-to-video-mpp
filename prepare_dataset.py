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
from futils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_train_model, train_options, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict, save_checkpoint
from futils.inference_utils import load_model as fu_load_model
from futils import hparams, audio
import warnings
warnings.filterwarnings("ignore")

args = train_options()
hparams = hparams.hparams

lnet_T = 5
# Weird function
def get_frame_id(self, frame):
    return int(basename(frame).split('.')[0])

def read_video(dataset, index, args):
    video_stream = cv2.VideoCapture(dataset[index])
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    if os.path.isfile(dataset[index].split('.')[0] + "_cropped.npy"):
        full_frames = np.load(dataset[index].split('.')[0] + "_cropped.npy")
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
    return full_frames


def get_segmented_mels(dataset, spec, start_frame):
    mels = []
    syncnet_mel_step_size = 16
    if start_frame >= 0: return None
    for i in range(start_frame, start_frame + lnet_T):
        m = self.crop_audio_window(spec, i - 2)
        if m.shape[0] != syncnet_mel_step_size:
            return None
        mels.append(m.T)
    mels = np.asarray(mels)
    return mels


def get_segmented_phones(self, index, start_frame):
    assert lnet_T == 5
    if start_frame < 1: return None
    # Get folder and file without ext.
    basefile = self.all_videos[index].split('.')[0]
    with open(basefile + ".json", 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    # Get Phones and words from json
    self.words = json_data['tiers']['words']['entries']
    # self.phones = json_data['tiers']['phones']

    m_fps = 1. / self.args.fps
    text_array = []
    for i in range(lnet_T):
        tmin = (start_frame + i) * m_fps
        tmax = (start_frame + i + 1) * m_fps
        tmp_word = []
        for (ts, te, word) in self.words:
            if ts < tmax and te >= tmin:
                tmp_word.append(word)
        text_array.append(" ".join(tmp_word))
    with torch.no_grad():
        text_tokens = clip.tokenize(text_array).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)
    return text_features


def crop_audio_window(self, spec, start_frame):
    syncnet_mel_step_size = 16
    start_idx = int(80. * (start_frame / float(hparams.fps)))
    end_idx = start_idx + syncnet_mel_step_size
    return spec[start_idx: end_idx, :]


def landmarks_estimate(self, nframes, save=False, start_frame=0):
    # face detection & cropping, cropping the first frame as the style of FFHQ
    croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')
    full_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in nframes]
    try:
        full_frames_RGB, crop, quad = croper.crop(full_frames_RGB, xsize=512)  # Why 512 ?
    except TypeError:
        return 1
    return 0

    clx, cly, crx, cry = crop
    lx, ly, rx, ry = quad
    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
    oy1, oy2, ox1, ox2 = cly + ly, min(cly + ry, nframes[0].shape[0]), clx + lx, min(clx + rx, nframes[0].shape[1])
    self.coordinates = oy1, oy2, ox1, ox2
    # original_size = (ox2 - ox1, oy2 - oy1)
    self.frames_pil = [Image.fromarray(cv2.resize(frame, (256, 256))) for frame in full_frames_RGB]

    # get the landmark according to the detected face.
    # Change this one
    if not os.path.isfile(self.all_videos[self.idx].split('.')[0] + '_landmarks.txt') or save:
        torch.cuda.empty_cache()
        # print('[Step 1] Landmarks Extraction in Video.')
        if self.kp_extractor is None:
            self.kp_extractor = KeypointExtractor()
        if not save:
            self.lm = self.kp_extractor.extract_keypoint(self.frames_pil)
        else:
            print("Save")
            self.lm = self.kp_extractor.extract_keypoint(self.frames_pil,
                                                         self.all_videos[self.idx].split('.')[0] + '_landmarks.txt')
    else:
        # print('[Step 1] Using saved landmarks.')
        self.lm = np.loadtxt(self.all_videos[self.idx].split('.')[0] + '_landmarks.txt').astype(np.float32)
        self.lm = self.lm[start_frame:start_frame + 2 * lnet_T]
        self.lm = self.lm.reshape([lnet_T, -1, 2])


def face_3dmm_extraction(self, save=False, start_frame=0):
    torch.cuda.empty_cache()
    if not os.path.isfile(self.all_videos[self.idx].split('.')[0] + "_coeffs.npy"):
        if self.net_recon is None:
            self.net_recon = load_face3d_net(self.args.face3d_net_path, device)
        lm3d_std = load_lm3d('checkpoints/BFM')
        video_coeffs = []
        for idx in range(len(self.frames_pil)):  # , desc="[Step 2] 3DMM Extraction In Video:"):
            frame = self.frames_pil[idx]
            W, H = frame.size
            lm_idx = self.lm[idx].reshape([-1, 2])
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
                coeffs = split_coeff(self.net_recon(im_idx_tensor))

            pred_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}
            pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'], \
                                         pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
            video_coeffs.append(pred_coeff)
        self.semantic_npy = np.array(video_coeffs)[:, 0]
        if save:
            np.save(self.all_videos[self.idx].split('.')[0] + '_coeffs.npy', self.semantic_npy)
    else:
        self.semantic_npy = np.load(self.all_videos[self.idx].split('.')[0] + "_coeffs.npy").astype(np.float32)
        self.semantic_npy = self.semantic_npy[start_frame:start_frame + lnet_T]


def hack_3dmm_expression(self, save=False, start_frame=0):
    expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_center'])[0]

    # Video Image Stabilized
    if not os.path.isfile(self.all_videos[self.idx].split('.')[0] + '_stablized.npy'):
        self.imgs = []
        for idx in range(len(self.frames_pil)):  # desc="[Step 3] Stablize the expression In Video:"):
            if self.args.one_shot:
                source_img = trans_image(self.frames_pil[0]).unsqueeze(0).to(device)
                semantic_source_numpy = self.semantic_npy[0:1]
            else:
                source_img = trans_image(self.frames_pil[idx]).unsqueeze(0).to(device)
                semantic_source_numpy = self.semantic_npy[idx:idx + 1]
            ratio = find_crop_norm_ratio(semantic_source_numpy, self.semantic_npy)
            coeff = transform_semantic(self.semantic_npy, idx, ratio).unsqueeze(0).to(device)

            # hacking the new expression
            coeff[:, :64, :] = expression[None, :64, None].to(device)
            with torch.no_grad():
                output = self.D_Net(source_img, coeff)
            img_stablized = np.uint8 \
                ((output['fake_image'].squeeze(0).permute(1, 2, 0).cpu().clamp_(-1, 1).numpy() + 1) / 2. * 255)
            self.imgs.append(cv2.cvtColor(img_stablized, cv2.COLOR_RGB2BGR))
        if save:
            np.save(self.all_videos[self.idx].split('.')[0] + '_stablized.npy', self.imgs)
        # del D_Net, model
        torch.cuda.empty_cache()
    else:
        self.imgs = np.load(self.all_videos[self.idx].split('.')[0] + "_stablized.npy")
        self.imgs = self.imgs[start_frame:start_frame + lnet_T]

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

            yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch
            img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch, ref_batch  = [], [], [], [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
        img_masked = img_batch.copy()
        img_original = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0
        img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch

def preprocess(dataset, args):
    # if not os.path.isfile(self.all_videos[self.idx].split('.')[0] +'_cropped.npy'):
    for idx in tqdm(range(len(dataset))):
        full_frames = read_video(dataset, idx, args)
        landmarks_estimate(full_frames, save=False)
        face_3dmm_extraction(save=False)
        hack_3dmm_expression(save=False)
        mel_chunks = get_full_mels()
        imgs_enhanced = get_enhanced_imgs()

        # Recrop face
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
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    filenames = get_image_list(args.data_root, 'train')
    seed = 42
    train_list, val_list = train_test_split(np.array(filenames), random_state=seed, train_size=0.8, test_size=0.2)
    print(len(filenames), len(train_list), len(val_list))
    # Dataset and Dataloader setup
    preprocess(train_list, args)
    preprocess(val_list, args)

if __name__ == "__main__":
    main()