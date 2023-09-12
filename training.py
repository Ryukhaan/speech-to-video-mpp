import glob

import gc
import torch
import torchvision

import numpy as np
import cv2, os, sys, subprocess, platform, torch
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat

from pydub import AudioSegment
from pydub.utils import make_chunks

import matplotlib.pyplot as plt
from scipy.signal import correlate

# Audio framework and Encodec
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio

import pickle
sys.path.append('third_part')
# 3dmm extraction
from third_part.face3d.util.preprocess import align_img
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.extract_kp_videos import KeypointExtractor
# face enhancement
from third_part.GPEN.face_enhancement import FaceEnhancement
from third_part.GFPGAN.gfpgan import GFPGANer
# expression control
from third_part.ganimation_replicate.model.ganimation import GANimationModel

from futils import audio
from futils.ffhq_preprocess import Croper
from futils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from futils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_train_model, options, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict, save_checkpoint
import warnings
warnings.filterwarnings("ignore")

args = options()

class ArcFaceLoss(torch.nn.Module):
    def __init__(self, device):
        super(ArcFaceLoss, self).__init__()
        self.face3d_net_path = 'checkpoints/face3d_pretrain_epoch_20.pth'
        self.device = device
        self.lm3d = 'checkpoints/BFM'
        self.l2_loss = torch.nn.MSELoss()

    def forward(self, y_pred, y_true):
        torch.cuda.empty_cache()
        net_recon = load_face3d_net(self.face3d_net_path, self.device)
        lm3d_std = load_lm3d(self.lm3d)

        W, H = y_pred.size
        lm_idx = lm[idx].reshape([-1, 2])
        if np.mean(lm_idx) == -1:
            lm_idx = (lm3d_std[:, :2]+1) / 2.
            lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
        else:
            lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

        # Y Predicted
        trans_params, im_idx, lm_idx, _ = align_img(y_pred, lm_idx, lm3d_std)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_idx_tensor = torch.tensor(np.array(im_idx)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
        with torch.no_grad():
            coeffs = split_coeff(net_recon(im_idx_tensor))
        pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
        pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'],\
                                         pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
        # Y Targets
        trans_params, im_idx, lm_idx, _ = align_img(y_true, lm_idx, lm3d_std)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_idx_tensor = torch.tensor(np.array(im_idx) / 255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
        with torch.no_grad():
            coeffs = split_coeff(net_recon(im_idx_tensor))
        true_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}
        true_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'], \
                                     pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
        return self.l2_loss(pred_coeff, true_coeff)

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval().to('cuda'))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval().to('cuda'))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval().to('cuda'))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval().to('cuda'))
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to('cuda'))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to('cuda'))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        #input = (input-self.mean) / self.std
        #target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input.to('cuda')
        y = target.to('cuda')
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class LNetLoss(torch.nn.Module):
    def __init__(self):
        super(LNetLoss, self).__init__()

    def forward(self, y_pred, y_true):

        y_pred = torchvision.transforms.Resize((384, 384))(y_pred)
        L1 = torch.nn.L1Loss()
        l1_val = L1(y_pred, y_true)

        L_perceptual = VGGPerceptualLoss()
        lp_val = L_perceptual(y_pred, y_true)

        # L_sync = 0.0
        lsync_val = 0.0  # L_sync(y_pred, y_true)

        lambda_1 = 1.
        lambda_p = 1.
        lambda_sync = 0.3
        return lambda_1 * l1_val + lambda_p * lp_val + lambda_sync * lsync_val

class ENetLoss(torch.nn.Module):
    def __init__(self, device):
        super(ENetLoss, self).__init__()
        self.device = device

    def forward(self, y_pred, y_true):

        # L1-Loss
        L1 = torch.nn.L1Loss()
        l1_val = L1(y_pred, y_true)

        # Loss Perceptual with VGG pretrained
        L_perceptual = VGGPerceptualLoss()
        lp_val = L_perceptual(y_pred, y_true)

        # Acrface loss (L2 function) for Identity
        l_id = ArcFaceLoss(self.device)
        lid_val = l_id(y_pred, y_true)

        # Adversial network AV-hubert ?
        #l_adv = ()
        #ladv_val = l_adv(y_pred, y_true)
        ladv_val = 0.
        # TODO : implement advsersial loss and id arcface loss

        lambda_1 = 0.2
        lambda_p = 1.
        lambda_adv = 100.
        lambda_id = 0.4
        return lambda_1 * l1_val + lambda_p * lp_val + lambda_adv * ladv_val + lambda_id * lid_val

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gc.collect()
    torch.cuda.empty_cache()
    print('[Info] Using {} for inference.'.format(device))

    base_name = args.face.split('/')[-1]
    # Image or Video ?
    if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')
    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

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
    # Encodec audio
    audio_encodec_model = EncodecModel.encodec_model_24khz()
    audio_encodec_model.set_target_bandwidth(12.0)
    wav, sr = torchaudio.load(args.audio)
    print(type(wav), wav.shape, sr)
    chunk_length_ms = 1  # pydub calculates in millisec
    for i in tqdm(range(sr, wav.shape[1], chunk_length_ms * sr)):
        chunk = wav[:, i * sr:(i + 1) * sr]
        print(chunk.size)
        chunk = convert_audio(chunk,
                              sr, audio_encodec_model.sample_rate, audio_encodec_model.channels)
        chunk = chunk.unsqueeze(0)
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = audio_encodec_model.encode(chunk)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
        print(codes.shape)
    exit()

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
    optimizer_LNet = torch.optim.Adam(model.parameters(), lr=0.001)
    #lnet_criterion = LNetLoss()
    enet_criterion = ENetLoss(device=device)
    for i, (img_batch, mel_batch, frames, coords, img_original, f_frames) in enumerate(
            tqdm(gen, desc='[Step 6] Lip Synthesis:',
                 total=int(np.ceil(float(len(mel_chunks)) / args.LNet_batch_size)))):

        optimizer_LNet.zero_grad()

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

        loss_E = enet_criterion(pred, reference)
        loss_E.required_grad = True
        loss_E.backward()

        #optimizer_LNet.step()
        optimizer_ENet.step()
    save_checkpoint(args.ENet_path + "_test.pth", model)

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

if __name__ == "__main__":
    train()
