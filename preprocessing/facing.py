import os
import cv2
import numpy as np
import sys
import subprocess
import platform
import torch
import gc
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat

import json
import os
from scipy.io import wavfile

sys.path.append('third_part')
# 3dmm extraction
from third_part.face3d.util.preprocess import align_img
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.extract_kp_videos import KeypointExtractor
# face enhancement
#from third_part.GPEN.face_enhancement import FaceEnhancement
#from third_part.GFPGAN.gfpgan import GFPGANer
# expression control
#from third_part.ganimation_replicate.model.ganimation import GANimationModel

from futils.ffhq_preprocess import Croper
from futils import audio
from futils.ffhq_preprocess import Croper
from futils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from futils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, load_lora_model, options, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gc.collect()
torch.cuda.empty_cache()
print('[Info] Using {} for inference.'.format(device))

class Preprocessor():

    def __init__(self, args):
        #super().__init__()
        self.args = args
        self.base_name = self.args.face.split('/')[-1] if args is not None else None
        self.full_frames = []
    def reading_video(self):

        # Image or Video ?
        if os.path.isfile(self.args.face) and self.args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            self.args.static = True
        if not os.path.isfile(self.args.face):
            raise ValueError('--face argument must be a valid path to video/image file')
        elif self.args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            img = cv2.imread(self.args.face)
            if img.shape[0] % 2 == 1:
                img = img[:-1, :, :]
            if img.shape[1] % 2 == 1:
                img = img[:, :-1, :]
            self.full_frames = [img, img]
            self.fps = self.args.fps
        else:
            video_stream = cv2.VideoCapture(self.args.face)
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
        print ("[Step 0] Number of frames available for inference:  " +str(len(self.full_frames)))

    def landmarks_estimate(self):
        # face detection & cropping, cropping the first frame as the style of FFHQ
        if not os.path.isfile(self.all_videos[self.idx].split('.')[0] + '_cropped.npy'):
            croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')
            full_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in self.full_frames]
            full_frames_RGB, crop, quad = croper.crop(full_frames_RGB, xsize=512) # Why 512 ?

            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly +ly, min(cly +ry, self.full_frames[0].shape[0]), clx +lx, min(clx +rx, self.full_frames[0].shape[1])
            self.coordinates = oy1, oy2, ox1, ox2
            # original_size = (ox2 - ox1, oy2 - oy1)
            self.frames_pil = [Image.fromarray(cv2.resize(frame ,(256 ,256))) for frame in full_frames_RGB]
            np.save('temp/'+ self.base_name + '_cropped.npy', np.array(self.frames_pil))
            np.save('temp/' + self.base_name + '_coordinates.npy', np.array(self.coordinates))
        else:
            self.coordinates = np.load('temp/' + self.base_name + '_coordinates.npy', allow_pickle=True)
            self.frames_pil = np.load('temp/'+ self.base_name + '_cropped.npy', allow_pickle=True)
            #self.frames_pil = [np.array(frame) for frame in self.frames_pil]
            #self.frames_96pil = np.asarray([cv2.resize(frame, (96, 96)) for frame in self.frames_pil])

        # get the landmark according to the detected face.
        if not os.path.isfile('temp/' + self.base_name + '_landmarks.txt') or self.args.re_preprocess:
            torch.cuda.empty_cache()
            print('[Step 1] Landmarks Extraction in Video.')
            kp_extractor = KeypointExtractor()
            self.lm = kp_extractor.extract_keypoint(self.frames_pil, './temp/ ' + self.base_name +'_landmarks.txt')
        else:
            print('[Step 1] Using saved landmarks.')
            self.lm = np.loadtxt('temp/' + self.base_name +'_landmarks.txt').astype(np.float32)
            self.lm = self.lm.reshape([len(self.full_frames), -1, 2])

    def face_3dmm_extraction(self):
        if not os.path.isfile('temp/' + self.base_name + '_coeffs.npy') \
            or self.args.exp_img is not None \
            or self.args.re_preprocess:
            torch.cuda.empty_cache()
            net_recon = load_face3d_net(self.args.face3d_net_path, device)
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
                    coeffs = split_coeff(net_recon(im_idx_tensor))

                pred_coeff = {key :coeffs[key].cpu().numpy() for key in coeffs}
                pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'], \
                                             pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
                video_coeffs.append(pred_coeff)
            self.semantic_npy = np.array(video_coeffs)[: ,0]
            np.save('temp/' + self.base_name +'_coeffs.npy', self.semantic_npy)
            del net_recon
        else:
            print('[Step 2] Using saved coeffs.')
            self.semantic_npy = np.load('temp/' + self.base_name +'_coeffs.npy').astype(np.float32)
    def hack_3dmm_expression(self):
        net_recon = load_face3d_net(self.args.face3d_net_path, device)

        # generate the 3dmm coeff from a single image
        if self.args.exp_img is not None \
                and ('.png' in self.args.exp_img or '.jpg' in self.args.exp_img):
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
                expression = split_coeff(net_recon(im_exp_tensor))['exp'][0]
        elif self.args.exp_img == 'smile':
            expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_mouth'])[0]
        else:
            print('using expression center')
            expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_center'])[0]
        del net_recon
        # load DNet, model(LNet and ENet)
        torch.cuda.empty_cache()
        if self.args.use_lora:
            self.D_Net, self.model = load_lora_model(self.args, device)
        else:
            self.D_Net, self.model = load_model(self.args, device)
        # Video Image Stabilized
        out = cv2.VideoWriter('temp/{}/stabilized.mp4'.format(self.args.tmp_dir),
                              cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (256, 256))

        if not os.path.isfile('temp/' + self.base_name +'_stablized.npy') or self.args.re_preprocess:
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

                out.write(cv2.cvtColor(img_stablized, cv2.COLOR_RGB2BGR))
            np.save('temp/' + self.base_name + '_stablized.npy', self.imgs)
        else:
            print('[Step 3] Using saved stablized video.')
            self.imgs = np.load('temp/' + self.base_name + '_stablized.npy')

        #del D_Net, model
        torch.cuda.empty_cache()

    def load_phones_dictionary(self):
        # Load Phones Dictionary
        with open(self.args.phones_dict_path, 'r') as file:
            self.dictionary = json.load(file)
        self.dictionary = self.dictionary['phones']
        self.dictionary.insert(0, 'spn')
        return self.dictionary

    def get_phones_per_ms(self):
        # Get folder and file without ext.
        folder = os.path.basename(self.args.json_path)
        basename_without_ext = os.path.splitext(os.path.basename(self.args.json_path))[0]
        with open(self.args.json_name, 'r') as file:
            json_data = json.load(file)

        # Get Phones and words from json
        words = json_data['tiers']['words']
        self.phones = json_data['tiers']['phones']

        # Load File WAV associated to the JSON
        wavfile = os.path.join([folder, basename_without_ext + '.wav'])
        samplerate, wav_data = wavfile.read(wavfile, 'r')
        milliseconds = len(wav_data) / samplerate * 1000

        # Each phones = (start_in_s, end_in_s, phone_str)
        self.phones_per_ms = np.zeros((int(milliseconds)+200, 1), dtype=np.int32)
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
            self.phones_per_ms[100+int(1000 * start):int(1000 * end)] = self.dictionary.index(phone)
        return self.phones_per_ms
