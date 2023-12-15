import glob

import gc
import torch
#from dtaidistance import dtw
from fastdtw import fastdtw

import numpy as np
import cv2, os, sys, subprocess, platform, torch
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal import correlate

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
from futils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, options, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict
import warnings
warnings.filterwarnings("ignore")

args = options()

import preprocessing.facing as preprocessing

def main():    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gc.collect()
    torch.cuda.empty_cache()
    print('[Info] Using {} for inference.'.format(device))
    preprocessor = preprocessing.Preprocessor(args)
    preprocessor.reading_video()
    preprocessor.landmarks_estimate()
    preprocessor.face_3dmm_extraction()
    preprocessor.hack_3dmm_expression()

    frames_pil = preprocessor.frames_pil
    full_frames = preprocessor.full_frames
    fps = preprocessor.fps
    imgs = preprocessor.imgs
    lm = preprocessor.lm
    oy1, oy2, ox1, ox2 = preprocessor.coordinates

    if not args.audio.endswith('.wav'):
        command = 'ffmpeg -loglevel error -y -i {} -strict -2 {}'.format(args.audio, 'temp/{}/temp.wav'.format(args.tmp_dir))
        subprocess.call(command, shell=True)
        args.audio = 'temp/{}/temp.wav'.format(args.tmp_dir)
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_step_size, mel_idx_multiplier, i, mel_chunks = 16, 80./fps, 0, []
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    #mel_chunks = mel_chunks[:4] # Change here length of inference video
    print("[Step 4] Load audio; Length of mel chunks: {}".format(len(mel_chunks)))
    imgs = imgs[:len(mel_chunks)]
    full_frames = full_frames[:len(mel_chunks)]  
    lm = lm[:len(mel_chunks)]

    #enhancer = FaceEnhancement(base_dir='checkpoints', size=1024, model='GPEN-BFR-1024', use_sr=False, \
    #                           sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)
    ref_enhancer = FaceEnhancement(args, base_dir='checkpoints',
                               in_size=512, channel_multiplier=2, narrow=1, sr_scale=4,
                               model='GPEN-BFR-512', use_sr=False)
    enhancer = FaceEnhancement(args, base_dir='checkpoints',
                               in_size=2048, channel_multiplier=2, narrow=1, sr_scale=2,
                               sr_model=None,
                               model='GPEN-BFR-2048', use_sr=True)

    imgs_enhanced = []
    for idx in tqdm(range(len(imgs)), desc='[Step 5] Reference Enhancement'):
        img = imgs[idx]
        #pred, _, _ = enhancer.process(img, aligned=True)
        pred, _, _ = ref_enhancer.process(img, img, face_enhance=False, possion_blending=False) #True
        imgs_enhanced.append(pred)
    gen = datagen(imgs_enhanced.copy(), mel_chunks, full_frames, None, (oy1,oy2,ox1,ox2))

    del ref_enhancer
    torch.cuda.empty_cache()

    frame_h, frame_w = full_frames[0].shape[:-1]
    if not args.cropped_image:
        out = cv2.VideoWriter('temp/{}/result.mp4'.format(args.tmp_dir), cv2.VideoWriter_fourcc(*'mp4v'), fps, (2*frame_w, 2*frame_h))
    else:
        out = cv2.VideoWriter('temp/{}/result.mp4'.format(args.tmp_dir), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (frame_w, frame_h))
    if args.up_face != 'original':
        instance = GANimationModel()
        instance.initialize()
        instance.setup()

    restorer = GFPGANer(model_path='checkpoints/GFPGANv1.4.pth', upscale=1, arch='clean', \
                        channel_multiplier=2, bg_upsampler=None)

    kp_extractor = KeypointExtractor()
    idx = 0
    for i, (img_batch, mel_batch, frames, coords, img_original, f_frames) in enumerate(tqdm(gen, desc='[Step 6] Lip Synthesis:', total=int(np.ceil(float(len(mel_chunks)) / args.LNet_batch_size)))):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        img_original = torch.FloatTensor(np.transpose(img_original, (0, 3, 1, 2))).to(device)/255. # BGR -> RGB
        
        with torch.no_grad():
            incomplete, reference = torch.split(img_batch, 3, dim=1) 
            pred, low_res = preprocessor.model(mel_batch, img_batch, reference)
            pred = torch.clamp(pred, 0, 1)

            if args.up_face in ['sad', 'angry', 'surprise']:
                tar_aus = exp_aus_dict[args.up_face]
            else:
                pass
            
            if args.up_face == 'original':
                cur_gen_faces = img_original
            else:
                test_batch = {'src_img': torch.nn.functional.interpolate((img_original * 2 - 1), size=(128, 128), mode='bilinear'), 
                              'tar_aus': tar_aus.repeat(len(incomplete), 1)}
                instance.feed_batch(test_batch)
                instance.forward()
                cur_gen_faces = torch.nn.functional.interpolate(instance.fake_img / 2. + 0.5, size=(384, 384), mode='bilinear')
                
            if args.without_rl1 is not False:
                incomplete, reference = torch.split(img_batch, 3, dim=1)
                mask = torch.where(incomplete==0, torch.ones_like(incomplete), torch.zeros_like(incomplete)) 
                pred = pred * mask + cur_gen_faces * (1 - mask) 
        
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        torch.cuda.empty_cache()
        delta = 0
        for p, f, xf, c in zip(pred, frames, f_frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            
            ff = xf.copy() 
            ff[y1:y2, x1:x2] = p
            
            # month region enhancement by GFPGAN
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                ff, has_aligned=False, only_center_face=True, paste_back=True)
                # 0,   1,   2,   3,   4,   5,   6,   7,   8,  9, 10,  11,  12,
            #mm =  [0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
            mm = [0,   0,   0,   0,   0,   0,   0,   0,   0,  0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
            mouse_mask = np.zeros_like(restored_img)
            tmp_mask = enhancer.faceparser.process(restored_img[y1:y2, x1:x2], mm)[0]
            #enhancer.faceparser.process(restored_img[y1:y2, x1:x2], mm)[0]
            mouse_mask[y1:y2, x1:x2]= cv2.resize(tmp_mask, (x2 - x1, y2 - y1))[:, :, np.newaxis] / 255.

            height, width = ff.shape[:2]
            restored_img, ff, full_mask = [cv2.resize(x, (512, 512)) for x in (restored_img, ff, np.float32(mouse_mask))]
            img = Laplacian_Pyramid_Blending_with_mask(restored_img, ff, full_mask[:, :, 0], 10)
            pp = np.uint8(cv2.resize(np.clip(img, 0 ,255), (width, height)))

            delta+=1
            if args.cropped_image:
                #pp, orig_faces, enhanced_faces = enhancer.process(pp, aligned=False)
                tmp_xf = cv2.resize(xf, (0,0), fx=2, fy=2)
                pp, orig_face, enhanced_faces = enhancer.process(pp, tmp_xf, bbox=c, face_enhance=True, possion_blending=True) # face=False
                pp = cv2.resize(pp, (0,0), fx=0.5, fy=0.5)
                ff = xf.copy()
                #ff[y1:y2, x1:x2] = pp[y1:y2, x1:x2]

                mask = np.zeros_like(ff)
                inverse_scale_x = (ox2 - ox1) / np.array(preprocessor.frames_pil[idx]).shape[1]
                inverse_scale_y = (oy2 - oy1) / np.array(preprocessor.frames_pil[idx]).shape[0]
                #dst_pts = lm[idx][-19:-1]
                #for j, (x,y) in enumerate(lm[idx]):
                #    xi, yi = int(inverse_scale_x * x + ox1), int(inverse_scale_y * y + oy1)
                #    cv2.circle(mask, (xi,yi), 3, (0,255,0), 1)
                #    cv2.putText(mask, str(j), (xi+5,yi), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

                nose = lm[idx][27:35+1]
                nose_mask = np.zeros_like(ff)
                element = np.ones((3,3), dtype=np.uint8)
                # Create Nose Mask
                for j, (x,y) in enumerate(nose):
                    xi, yi = int(inverse_scale_x*x + x1), int(512-inverse_scale_y*y + y1)
                    xj, yj = int(inverse_scale_x*nose[j-1][0] + x1), int(512-inverse_scale_y*nose[j-1][1] + y1)
                    cv2.line(nose_mask, (xj, yj), (xi, yi), (255,0,0), 3)
                nose_mask = nose_mask[:,:,0].astype(np.uint8)
                # Imfill nose mask
                h, w = nose_mask.shape[:2]
                fill_mask = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(nose_mask, fill_mask, (0, 0), 255)
                nose_mask = cv2.bitwise_not(nose_mask)
                # Dilate to have less incoherence
                nose_mask = cv2.dilate(nose_mask, element, iterations=10)

                # Draw bottom face
                bottom_face = lm[idx][0:16 + 1]
                for j, (x,y) in enumerate(bottom_face):
                    xi, yi = int(inverse_scale_x*x + ox1), int(512-inverse_scale_y*y + oy1)
                    xj, yj = int(inverse_scale_x*bottom_face[j - 1][0] + ox1), int(512-inverse_scale_y*bottom_face[j - 1][1] + oy1)
                    cv2.line(mask, (xj, yj), (xi,yi), (255,0,0), 2)
                # Filled
                mask = mask[:, :, 0].astype(np.uint8)
                fill_mask = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(mask, fill_mask, (0, 0), 255)
                mask = cv2.bitwise_not(mask)

                # Remove nose from bottom face
                mask = np.multiply(mask, 1 - nose_mask)
                # Apply to each channel
                #cv2.imwrite("./results/full_mask{}.png".format(idx), mask)
                for channel in range(ff.shape[2]):
                    ff_masked = np.multiply(ff[:,:,channel], np.logical_not(mask))
                    pp_masked = np.multiply(pp[:,:,channel], mask>0)
                    ff[:,:,channel] = ff_masked + pp_masked

                # Visual debug
                #ff = cv2.rectangle(ff, (ox1, oy1), (ox2, oy2), (255,0,0))
                #cv2.circle(ff, (ox1, oy1), 3, (0,255,0), 1)
                #cv2.circle(ff, (ox2, oy2), 3, (0,0,255), 1)
                # Draw detected mouth landmarks
                mouth = lm[idx][48:]
                for j, (x,y) in enumerate(mouth):
                    xi, yi = int(inverse_scale_x*x + ox1), int(512-inverse_scale_y*y+oy1)
                    cv2.circle(ff, (xi, yi), 3, (255, 0, 0), 1)
                for j, (x, y) in enumerate(bottom_face):
                    xi, yi = int(inverse_scale_x*x + ox1), int(512-inverse_scale_y*y+oy2)
                    cv2.circle(ff, (xi, yi), 3, (255, 0, 0), 1)
                assert ff.shape[0] == frame_h and ff.shape[1] == frame_w, print(ff.shape, frame_h, frame_w)
                #cv2.imwrite("./results/{}.png".format(idx), ff)
                out.write(ff)
                idx += 1
            else:
                tmp_xf = cv2.resize(xf, (0, 0), fx=2, fy=2)
                pp, orig_faces, enhanced_faces = enhancer.process(pp, tmp_xf, bbox=c, face_enhance=True, possion_blending=True)
                out.write(pp)
    out.release()
    
    if not os.path.isdir(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    command = 'ffmpeg -loglevel error -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/{}/result.mp4'.format(args.tmp_dir), args.outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')
    print('outfile:', args.outfile)


# frames:256x256, full_frames: original size
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
        #idx = 0 if args.static else i % len(frames) # HERE !!
        if args.static:
            idx = 0
        else:
            if i >= len(frames):
                idx = len(frames) - (i - len(frames)) - 1
            else:
                idx = i
        #idx = 0 if args.static else i if
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


def find_best_audio():

    # Get basename
    base_name = args.face.split('/')[-1]
    # Get model name
    model_name = args.face.split('/')[-2]
    print(model_name)
    if args.static is True:
        return
    # Create temp directory
    os.makedirs(os.path.join('temp', args.tmp_dir), exist_ok=True)
    # Make it again if args is passed or the best audio is not already found
    if not os.path.isfile('temp/'+base_name+'_best_audio.npy') or args.re_preprocess:
        # TODO : Try database by gender instead the same actor

        # Get audio database of the same actor
        audio_database = glob.glob('/'.join(args.audio.split('/')[:-1]) + '/*.wav')
        #audio_database = glob.glob(os.getcwd() + "/../../data/audio/antoine/*.wav")

        # Open the source wav file
        src_wav = np.double(audio.load_wav(args.audio, 16000))

        # Old version with L1-distance of mel spectogram
        #src_mel = audio.melspectrogram(src_wav)
        #_, src_length = src_mel.shape

        best_distance = np.inf
        best_audio = ""

        # Process distances of all audio in database
        for file in tqdm(audio_database, desc='[Step 0 bis] Finding best audio:'):
            if file == args.audio: continue
            dst_wav = np.double(audio.load_wav(file, 16000))

            # Old version with L1-distance and mel spectogram
            #dst_mel = audio.melspectrogram(dst_wav)
            #_, dst_length = dst_mel.shape
            #if dst_length >= src_length:
            #
            #    tmp_src_mel = np.pad(src_mel, ((0,0),(0,dst_length-src_length)))
            #    current_sim = np.mean(np.linalg.norm(tmp_src_mel - dst_mel, axis=1))

            current_distance, _ = fastdtw(src_wav, dst_wav)

            if current_distance < best_distance:
                best_audio = file
                best_distance = current_distance

        # Get the video corresponding to the best audio found
        best_vid = os.path.join("../../data/video/", model_name, best_audio.split('/')[-1][:-3] + 'mp4')
        args.face = best_vid
        np.save('temp/' + base_name + '_best_audio.npy', best_vid)
    else:
        args.face = str(np.load('temp/' + base_name + '_best_audio.npy'))
    print(args.face)

if __name__ == '__main__':
    #find_best_audio()
    main()
