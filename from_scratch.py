import os, sys

sys.path.append('third_part')

import argparse
from third_part.face3d.extract_kp_videos import KeypointExtractor
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

from futils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from futils.inference_utils import face_detect
from futils.ffhq_preprocess import Croper

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

def get_image_list(dirpath):
    filelist = []
    for folder in tqdm(os.listdir(dirpath), position=0, leave=True):
        if folder == '.DS_Store': continue
        for vidname in tqdm(os.listdir(dirpath + '/' + folder), position=1, leave=False):
            if vidname == '.DS_Store': continue
            if vidname.split('.')[-1] != 'mp4': continue
            filelist.append(os.path.join(dirpath, folder, vidname))
    return filelist

def read_video(vidname):
    video_stream = cv2.VideoCapture(vidname)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        y1, y2, x1, x2 = [0, -1, 0, -1]
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]
        full_frames.append(frame)
    return full_frames

args = options()

if __name__ == "__main__":
    vidnames = get_image_list(args.data_root)
    # original frames
    kp_extractor = KeypointExtractor()
    image_size = 256
    for file in tqdm(vidnames):
        full_frames = read_video(file)
        frames = full_frames.copy()

        # face detection & cropping, cropping the first frame as the style of FFHQ
        croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')
        full_frames_RGB = [np.asarray(frame) for frame in frames]

        # Why there was a try ?
        full_frames_RGB, crop, quad = croper.crop(full_frames_RGB, xsize=512)  # Why 512 ?

        clx, cly, crx, cry = crop
        lx, ly, rx, ry = quad
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        oy1, oy2, ox1, ox2 = cly + ly, min(cly + ry, full_frames_RGB[0].shape[0]), clx + lx, min(clx + rx,
                                                                                                 full_frames_RGB[
                                                                                                     0].shape[1])
        coordinates = oy1, oy2, ox1, ox2


        fr_pil = [Image.fromarray(frame) for frame in frames]
        lms = kp_extractor.extract_keypoint(fr_pil, 'temp/'+'temp_x12_landmarks.txt')
        frames_pil = [(lm, frame) for frame,lm in zip(fr_pil, lms)] # frames is the croped version of modified face
        crops, orig_images, quads  = crop_faces(image_size, frames_pil, scale=1.0, use_fa=True)
        inverse_transforms = [calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]) for quad in quads]


        face_det_results = face_detect(full_frames, args, jaw_correction=True)

        refs = []
        #_fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        #out = cv2.VideoWriter('./outpy.mp4', _fourcc, 25, (96, 96))
        #i = 0
        for inverse_transform, crop, full_frame, face_det in zip(inverse_transforms, crops, full_frames_RGB, face_det_results):
            imc_pil = paste_image(inverse_transform, crop, Image.fromarray(
                cv2.resize(full_frame[int(oy1):int(oy2), int(ox1):int(ox2)], (256, 256))))

            ff = full_frame.copy()
            #ff[int(oy1):int(oy2), int(ox1):int(ox2)] = cv2.resize(np.array(imc_pil.convert('RGB')), (ox2 - ox1, oy2 - oy1))
            oface, coords = face_det
            y1, y2, x1, x2 = coords
            #cv2.imwrite("./results/{}.png".format(i), ff[y1:y2, x1:x2])
            #i += 1
            refs.append(ff[y1: y2, x1:x2])
            #out.write(ff[y1:y2, x1:x2])
        print(file + "_img_batch.npy")
        np.save(file + '_img_batch.npy', refs)
        break
