import glob
from os.path import dirname, join, basename, isfile

import json
import gc
import torch
from torch import nn
from torch import optim
import torchvision
from torchsummary import summary
from torch.utils import data as data_utils

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
from futils import hparams
import warnings
warnings.filterwarnings("ignore")

args = train_options()

lnet_T = 5

def get_image_list(data_root, split):
    filelist = []
    with open('filelists/{}.txt'.format(split)) as f:
        for line in f:
            if line.split('.')[-1] == 'wav':
                filelist.append(os.path.join(data_root, line))
    return filelist

class Dataset(object):

    def __init__(self, filenames):
        global args
        self.args = args
        self.all_videos = filenames #get_image_list(args.data_root, split)
        self.preprocessor = preprocessing.Preprocessor(args=None)
        self.net_recon = None

    # Weird function
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def read_video(self, index):
        video_stream = cv2.VideoCapture(self.all_videos[index])
        self.fps = video_stream.get(cv2.CAP_PROP_FPS)
        self.full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            y1, y2, x1, x2 = self.args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            self.full_frames.append(frame)
        return self.full_frames

    def get_segmented_window(self, start_frame):
        assert lnet_T == 5
        if start_frame < 1: return None
        return self.full_frames[start_frame-2:start_frame+lnet_T-2]

    def get_segmented_codes(self, index, start_frame):
        assert lnet_T == 5
        if start_frame < 1: return None
        codes = np.load(basename(self.all_videos[index]).split('.')[0] + "_codes.npy",
                allow_pickle=True)
        codes = codes[start_frame-2: start_frame+lnet_T-2]
        codes = codes.reshape(-1, 32 * 15)
        print(codes.shape)
        return codes

    def get_segmented_phones(self, index, start_frame):
        assert lnet_T == 5
        if start_frame < 1: return None
        # Get folder and file without ext.
        basefile_name = basename(self.all_videos[index]).split('.')[0]
        with open(basefile_name + ".json", 'r') as file:
            json_data = json.load(file)

        # Get Phones and words from json
        words = json_data['tiers']['words']
        self.phones = json_data['tiers']['phones']

        # Load File WAV associated to the JSON
        samplerate, wav_data = wavfile.read(basefile_name + ".wav", 'r')
        milliseconds = len(wav_data) / samplerate * 1000

        # Each phones = (start_in_s, end_in_s, phone_str)
        self.phones_per_ms = np.zeros((int(milliseconds), 1), dtype=np.int32)
        for (start, end, phone) in self.phones['entries']:
            # Some errors have been transcribed by MFA
            if phone == "d̪":
                phone = "ð"
            if phone == "t̪":
                phone = "θ"
            if phone == "kʷ":
                phone = "k"
            if phone == "tʷ":
                phone = "t"
            if phone == "cʷ":
                phone = "k"
            if phone == "ɾʲ":
                phone = "ɒ"
            if phone == "ɾ̃":
                phone = "θ"
            if phone == "ɟʷ":
                phone = "ɟ"
            if phone == "ɡʷ":
                phone = "ɡ"
            if phone == "vʷ":
                phone = "v"
            self.phones_per_ms[int(1000 * start):int(1000 * end)] = self.dictionary.index(phone)
        self.phones_per_ms = np.append(self.phones_per_ms, (100, 100),'constant', constant_value=0)
        phones = self.phones_per_ms[100 + (start_frame-2)*200 : 100 + (start_frame-2+lnet_T)*200 ]
        print(phones.shape)
        return phones

    def prepare_window(self, window):
        # Convert to 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))
        return x

    def landmarks_estimate(self, nframes):
        # face detection & cropping, cropping the first frame as the style of FFHQ
        croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')
        full_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in nframes]
        full_frames_RGB, crop, quad = croper.crop(full_frames_RGB, xsize=512) # Why 512 ?

        clx, cly, crx, cry = crop
        lx, ly, rx, ry = quad
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        oy1, oy2, ox1, ox2 = cly +ly, min(cly +ry, nframes[0].shape[0]), clx +lx, min(clx +rx, nframes[0].shape[1])
        self.coordinates = oy1, oy2, ox1, ox2
        # original_size = (ox2 - ox1, oy2 - oy1)
        self.frames_pil = [Image.fromarray(cv2.resize(frame ,(256 ,256))) for frame in full_frames_RGB]

        # get the landmark according to the detected face.
        # Change this one
        #if not os.path.isfile('temp/ ' + self.base_name +'_landmarks.txt') or self.args.re_preprocess:
        torch.cuda.empty_cache()
        print('[Step 1] Landmarks Extraction in Video.')
        kp_extractor = KeypointExtractor()
        self.lm = kp_extractor.extract_keypoint(self.frames_pil, './temp/ ' + self.base_name +'_landmarks.txt')
        #else:
        #    print('[Step 1] Using saved landmarks.')
        #    self.lm = np.loadtxt('temp/ ' + self.base_name +'_landmarks.txt').astype(np.float32)
        #    self.lm = self.lm.reshape([len(self.full_frames), -1, 2])

    def face_3dmm_extraction(self):
        torch.cuda.empty_cache()
        if self.net_recon is None:
            self.net_recon = load_face3d_net(self.args.face3d_net_path, device)
        lm3d_std = load_lm3d('checkpoints/BFM')
        video_coeffs = []
        for idx in tqdm(range(len(self.frames_pil)), desc="[Step 2] 3DMM Extraction In Video:"):
            frame = self.frames_pil[idx]
            W, H = frame.size
            lm_idx = self.lm[idx].reshape([-1, 2])
            if np.mean(lm_idx) == -1:
                lm_idx = (lm3d_std[:, :2 ] +1) / 2.
                lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
            else:
                lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

            trans_params, im_idx, lm_idx, _ = align_img(frame, lm_idx, lm3d_std)
            trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
            im_idx_tensor = torch.tensor(np.array(im_idx ) /255., dtype=torch.float32).permute(2, 0, 1).to \
                (device).unsqueeze(0)
            with torch.no_grad():
                coeffs = split_coeff(self.net_recon(im_idx_tensor))

            pred_coeff = {key :coeffs[key].cpu().numpy() for key in coeffs}
            pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'], \
                                         pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
            video_coeffs.append(pred_coeff)
        self.semantic_npy = np.array(video_coeffs)[: ,0]
        #np.save('temp/ ' + self.base_name +'_coeffs.npy', self.semantic_npy)
        #del net_recon

    def hack_3dmm_expression(self):
        print('extract the exp from' , self.args.exp_img)
        exp_pil = Image.open(self.args.exp_img).convert('RGB')
        lm3d_std = load_lm3d('third_part/face3d/BFM')

        W, H = exp_pil.size
        kp_extractor = KeypointExtractor()
        lm_exp = kp_extractor.extract_keypoint([exp_pil], 'temp/ ' + self.base_name +'_temp.txt')[0]
        if np.mean(lm_exp) == -1:
            lm_exp = (lm3d_std[:, :2] + 1) / 2.
            lm_exp = np.concatenate(
                [lm_exp[:, :1] * W, lm_exp[:, 1:2] * H], 1)
        else:
            lm_exp[:, -1] = H - 1 - lm_exp[:, -1]

        trans_params, im_exp, lm_exp, _ = align_img(exp_pil, lm_exp, lm3d_std)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_exp_tensor = torch.tensor(np.array(im_exp ) /255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0)
        with torch.no_grad():
            expression = split_coeff(self.net_recon(im_exp_tensor))['exp'][0]

        torch.cuda.empty_cache()
        self.D_Net, self.model = fu_load_model(self.args, device)

        # Video Image Stabilized
        self.imgs = []
        for idx in tqdm(range(len(self.frames_pil)), desc="[Step 3] Stablize the expression In Video:"):
            if self.args.one_shot:
                source_img = trans_image(self.frames_pil[0]).unsqueeze(0).to(device)
                semantic_source_numpy = self.semantic_npy[0:1]
            else:
                source_img = trans_image(self.frames_pil[idx]).unsqueeze(0).to(device)
                semantic_source_numpy = self.semantic_npy[idx:idx +1]
            ratio = find_crop_norm_ratio(semantic_source_numpy, self.semantic_npy)
            coeff = transform_semantic(self.semantic_npy, idx, ratio).unsqueeze(0).to(device)

            # hacking the new expression
            coeff[:, :64, :] = expression[None, :64, None].to(device)
            with torch.no_grad():
                output = self.D_Net(source_img, coeff)
            img_stablized = np.uint8 \
                ((output['fake_image'].squeeze(0).permute(1 ,2 ,0).cpu().clamp_(-1, 1).numpy() + 1 )/ 2. * 255)
            self.imgs.append(cv2.cvtColor(img_stablized, cv2.COLOR_RGB2BGR))

        #del D_Net, model
        torch.cuda.empty_cache()
    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while True:
            idx = np.random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            frames = self.read_video(idx)
            # Sure that nframe if >= 2 and lower than N - 3
            start_frame = np.random.randint(2, len(frames) - 3)

            nframes = self.get_segmented_window(start_frame)
            codes  = self.get_segmented_codes(idx, start_frame)
            phones = self.get_segmented_phones(idx, start_frame)

            self.landmarks_estimate(nframes)
            self.face_3dmm_extraction()
            self.hack_3dmm_expression()




def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    loss_func = losses.LNetLoss()
    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, code, phone, y) in prog_bar:
            model.train()
            optimizer.zero_grad()

            x = x.to(device)
            mel = mel.to(device)
            y = y.to(device)

            pred = model(x, code, phone)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()
            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))
        global_epoch += 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    gc.collect()
    torch.cuda.empty_cache()
    print('[Info] Using {} for inference.'.format(device))
    preprocessor = preprocessing.Preprocessor(args)
    preprocessor.reading_video()

    # Select 5 frames
    full_frames = preprocessor.full_frames
    i = np.random.randint(0, high=len(full_frames)-5)
    full_frames = full_frames[i:i+5]
    preprocessor.full_frames = full_frames

    preprocessor.landmarks_estimate()
    preprocessor.face_3dmm_extraction()
    preprocessor.hack_3dmm_expression()

    # base_name = args.face.split('/')[-1]
    # # Image or Video ?
    # if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    #     args.static = True
    # if not os.path.isfile(args.face):
    #     raise ValueError('--face argument must be a valid path to video/image file')
    # elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    #     full_frames = [cv2.imread(args.face)]
    #     fps = args.fps
    # else:
    #     video_stream = cv2.VideoCapture(args.face)
    #     fps = video_stream.get(cv2.CAP_PROP_FPS)
    #
    #     full_frames = []
    #     while True:
    #         still_reading, frame = video_stream.read()
    #         if not still_reading:
    #             video_stream.release()
    #             break
    #         y1, y2, x1, x2 = args.crop
    #         if x2 == -1: x2 = frame.shape[1]
    #         if y2 == -1: y2 = frame.shape[0]
    #         frame = frame[y1:y2, x1:x2]
    #         full_frames.append(frame)

    print("[Step 0] Number of frames available for inference: " + str(len(full_frames)))
    # face detection & cropping, cropping the first frame as the style of FFHQ
    croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')
    full_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in full_frames]
    full_frames_RGB, crop, quad = croper.crop(full_frames_RGB, xsize=512)

    clx, cly, crx, cry = crop
    lx, ly, rx, ry = quad
    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
    oy1, oy2, ox1, ox2 = cly + ly, min(cly + ry, full_frames[0].shape[0]), clx + lx, min(clx + rx,
                                                                                         full_frames[0].shape[1])
    # original_size = (ox2 - ox1, oy2 - oy1)
    frames_pil = [Image.fromarray(cv2.resize(frame, (256, 256))) for frame in full_frames_RGB]

    # get the landmark according to the detected face.
    if not os.path.isfile('temp/' + base_name + '_landmarks.txt') or args.re_preprocess:
        torch.cuda.empty_cache()
        print('[Step 1] Landmarks Extraction in Video.')
        kp_extractor = KeypointExtractor()
        lm = kp_extractor.extract_keypoint(frames_pil, './temp/' + base_name + '_landmarks.txt')
    else:
        print('[Step 1] Using saved landmarks.')
        lm = np.loadtxt('temp/' + base_name + '_landmarks.txt').astype(np.float32)
        lm = lm.reshape([len(full_frames), -1, 2])

    print(not os.path.isfile('temp/' + base_name + '_coeffs.npy'), args.exp_img is not None, args.re_preprocess)
    if not os.path.isfile('temp/' + base_name + '_coeffs.npy') or args.exp_img is not None or args.re_preprocess:
        torch.cuda.empty_cache()
        net_recon = load_face3d_net(args.face3d_net_path, device)
        lm3d_std = load_lm3d('checkpoints/BFM')

        video_coeffs = []
        for idx in tqdm(range(len(frames_pil)), desc="[Step 2] 3DMM Extraction In Video:"):
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
            im_idx_tensor = torch.tensor(np.array(im_idx) / 255., dtype=torch.float32).permute(2, 0, 1).to(
                device).unsqueeze(0)
            with torch.no_grad():
                coeffs = split_coeff(net_recon(im_idx_tensor))

            pred_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}
            pred_coeff = np.concatenate(
                [pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'], \
                 pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
            video_coeffs.append(pred_coeff)
        semantic_npy = np.array(video_coeffs)[:, 0]
        np.save('temp/' + base_name + '_coeffs.npy', semantic_npy)
    else:
        print('[Step 2] Using saved coeffs.')
        semantic_npy = np.load('temp/' + base_name + '_coeffs.npy').astype(np.float32)

    # generate the 3dmm coeff from a single image
    if args.exp_img is not None and ('.png' in args.exp_img or '.jpg' in args.exp_img):
        print('extract the exp from', args.exp_img)
        exp_pil = Image.open(args.exp_img).convert('RGB')
        lm3d_std = load_lm3d('third_part/face3d/BFM')

        W, H = exp_pil.size
        kp_extractor = KeypointExtractor()
        lm_exp = kp_extractor.extract_keypoint([exp_pil], 'temp/' + base_name + '_temp.txt')[0]
        if np.mean(lm_exp) == -1:
            lm_exp = (lm3d_std[:, :2] + 1) / 2.
            lm_exp = np.concatenate(
                [lm_exp[:, :1] * W, lm_exp[:, 1:2] * H], 1)
        else:
            lm_exp[:, -1] = H - 1 - lm_exp[:, -1]

        trans_params, im_exp, lm_exp, _ = align_img(exp_pil, lm_exp, lm3d_std)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_exp_tensor = torch.tensor(np.array(im_exp) / 255., dtype=torch.float32).permute(2, 0, 1).to(
            device).unsqueeze(0)
        with torch.no_grad():
            expression = split_coeff(net_recon(im_exp_tensor))['exp'][0]
        del net_recon
    elif args.exp_img == 'smile':
        expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_mouth'])[0]
    else:
        print('using expression center')
        expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_center'])[0]

    # load DNet, model(LNet and ENet)
    torch.cuda.empty_cache()
    D_Net, L_Net, model =  load_train_model(args, device)
    model.set_training_style()
    summary(model)
    # Video Image Stabilized
    out = cv2.VideoWriter('temp/{}/stabilized.mp4'.format(args.tmp_dir),
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (256, 256))
    if not os.path.isfile('temp/' + base_name + '_stablized.npy') or args.re_preprocess:
        imgs = []
        for idx in tqdm(range(len(frames_pil)), desc="[Step 3] Stablize the expression In Video:"):
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
            img_stablized = np.uint8(
                (output['fake_image'].squeeze(0).permute(1, 2, 0).cpu().clamp_(-1, 1).numpy() + 1) / 2. * 255)
            imgs.append(cv2.cvtColor(img_stablized, cv2.COLOR_RGB2BGR))

            out.write(cv2.cvtColor(img_stablized, cv2.COLOR_RGB2BGR))
        np.save('temp/' + base_name + '_stablized.npy', imgs)
        del D_Net
    else:
        print('[Step 3] Using saved stablized video.')
        imgs = np.load('temp/' + base_name + '_stablized.npy')

    # return 0

    if not args.audio.endswith('.wav'):
        command = 'ffmpeg -loglevel error -y -i {} -strict -2 {}'.format(args.audio,
                                                                         'temp/{}/temp.wav'.format(args.tmp_dir))
        subprocess.call(command, shell=True)
        args.audio = 'temp/{}/temp.wav'.format(args.tmp_dir)
    # # Encodec audio
    # audio_encodec_model = EncodecModel.encodec_model_24khz()
    # audio_encodec_model.set_target_bandwidth(24.0)
    # wav, sr = torchaudio.load(args.audio)
    # print(sr)
    # t = 5
    # idx_multiplier, mel_chunks = int(1. / fps * sr), []
    # #for i, _ in enumerate(tqdm(range(0, wav.shape[1], idx_multiplier), total=int(wav.shape[1] / idx_multiplier))):
    # for i, _ in enumerate(tqdm(range(len(full_frames)-t))):
    #     chunk = wav[:, i*idx_multiplier:(i+t)*idx_multiplier]
    #     chunk = convert_audio(chunk,
    #                           sr, audio_encodec_model.sample_rate, audio_encodec_model.channels)
    #     chunk = chunk.unsqueeze(0)
    #     # Extract discrete codes from EnCodec
    #     with torch.no_grad():
    #         encoded_frames = audio_encodec_model.encode(chunk)
    #     codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
    #     mel_chunks.append(codes)
    # print(mel_chunks[0], mel_chunks[4*t], mel_chunks[-1].shape)
    # print("[Step 4 bis] Load audio; Length of mel chunks: {}".format(len(mel_chunks)))
    # exit()

    wav = audio.load_wav(args.audio, 16000)
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

    print("[Step 4] Load audio; Length of mel chunks: {}".format(len(mel_chunks)))
    imgs = imgs[:len(mel_chunks)]
    full_frames = full_frames[:len(mel_chunks)]
    lm = lm[:len(mel_chunks)]

    # enhancer = FaceEnhancement(base_dir='checkpoints', size=1024, model='GPEN-BFR-1024', use_sr=False, \
    #                           sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)
    enhancer = FaceEnhancement(args, base_dir='checkpoints', in_size=1024, model='GPEN-BFR-1024', use_sr=False)
    imgs_enhanced = []
    if not os.path.isfile('temp/' + base_name + '_enhanced5.npy') or args.re_preprocess:
        for idx in tqdm(range(len(imgs)), desc='[Step 5] Reference Enhancement'):
            img = imgs[idx]
            # pred, _, _ = enhancer.process(img, aligned=True)
            pred, _, _ = enhancer.process(img, img, face_enhance=True, possion_blending=False)
            imgs_enhanced.append(pred)
        np.save('temp/' + base_name + '_enhanced5.npy', imgs_enhanced)
    else:
        print('[Step 5] Using saved reference enhancement.')
        imgs_enhanced = np.load('temp/' + base_name + '_enhanced5.npy')

    if not os.path.isfile('temp/' + base_name + '_gen.npy') or args.re_preprocess:
        gen = datagen(imgs_enhanced.copy(), mel_chunks, full_frames, None, (oy1, oy2, ox1, ox2))
        l_gen = [(img_batch, mel_batch, frames, coords, img_original, f_frames) for (img_batch, mel_batch, frames, coords, img_original, f_frames) in gen]
        pickle.dump(l_gen, open('temp/' + base_name + '_gen.npy', 'wb'))
        #np.save('temp/' + base_name + '_gen.npy', gen)
    else:
        print('Using saved generator.')
        gen = pickle.load(open('temp/' + base_name + '_gen.npy', 'rb'))
        #gen = np.load('temp/' + base_name + '_gen.npy')
    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/{}/result.mp4'.format(args.tmp_dir), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (frame_w, frame_h))

    if args.up_face != 'original':
        instance = GANimationModel()
        instance.initialize()
        instance.setup()

    restorer = GFPGANer(model_path='checkpoints/GFPGANv1.4.pth', upscale=3, arch='clean', \
                        channel_multiplier=2, bg_upsampler=None)
    kp_extractor = KeypointExtractor()

    #optimizer_LNet = torch.optim.Adam(L_Net.parameters(), lr=0.001)
    optimizer_ENet = torch.optim.Adam(model.parameters(), lr=0.01)
    #lnet_criterion = LNetLoss()
    enet_criterion = ENetLoss(device=device)

    #torch.set_grad_enabled(True)
    for epoch in range(10):
        bar = tqdm(gen, desc='[Step 6] Lip Synthesis:',
                   total=int(np.ceil(float(len(mel_chunks)) / args.LNet_batch_size)))
        for i, (img_batch, mel_batch, frames, coords, img_original, f_frames) in enumerate(bar):

            optimizer_ENet.zero_grad()

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
            #img_original = torch.FloatTensor(np.transpose(img_original, (0, 3, 1, 2))).to(device) / 255.  # BGR -> RGB
            incomplete, reference = torch.split(img_batch, 3, dim=1)
            pred, low_res = model(mel_batch, img_batch, reference)

            #own_net.zero_grad()
            #t = own_net(reference)
            #t.requires_grad
            #reference.requires_grad

            #loss_L = lnet_criterion(pred, reference)
            #loss_L.requires_grad = True
            #loss_L.backward()

            #pred = torch.clamp(pred, 0, 1).to(device)
            #low_res = low_res.to(device)
            #reference = reference.to(device)
            #loss_L = torch.nn.L1Loss()(pred, reference)
            #loss_L.required_grad = True
            #loss_L.backward()
            print(type(pred), type(semantic_npy), type(semantic_npy[i]))
            loss_E = enet_criterion(semantic_npy[i], pred, reference)
            loss_E.requires_grad = True
            loss_E.backward()
            bar.set_description("{}".format(loss_E))
            #optimizer_LNet.step()
            optimizer_ENet.step()
        save_checkpoint(args.ENet_path + "_test_2{}.pth".format(epoch), model)

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

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    loss = losses.LNetLoss()
    while 1:
        for step, (x, codes, phones, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)
            codes = codes.to(device)
            phones = phones.to(device)

            pred = model(x, codes, phones)
            y = y.to(device)

            loss = loss(pred, y)
            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)

        return

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    filenames = get_image_list(args.data_root, 'train')
    seed = 42
    train_list, val_list = train_test_split(np.array(filenames), random_state=seed, train_size=0.7, test_size=0.3)
    print(len(filenames), len(train_list), len(val_list))
    # Dataset and Dataloader setup
    train_dataset = Dataset(train_list)
    test_dataset = Dataset(val_list)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = LNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)
