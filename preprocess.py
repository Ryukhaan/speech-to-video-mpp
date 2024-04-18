import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('checkpoints/s3fd.pth'):
    raise FileNotFoundError('Save the s3fd model to checkpoints/s3fd.pth \
							before running this script!')

import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import futils.audio as audio
from futils.hparams import hparams as hp

sys.path.append('third_part')
import third_part.face_detection as face_detection

import gc
import torch
from torchaudio import load as torch_load
from torchaudio.functional import resample
from encodec import EncodecModel
from encodec.utils import convert_audio
import clip as torch_clip

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)

parser.add_argument("--bandwidth", default=24.0, type=float, help="Bandwidth value (by default 24.0)")
parser.add_argument("--chunk_length_s", default=.2, type=float, help="Second which frame length")
parser.add_argument("--fps", help="Frame per second (default 25)", default=25, type=int)

args = parser.parse_args()

# Device Cuda or CPU and then set cache empty
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gc.collect()
torch.cuda.empty_cache()

# Load encodec
#configuration = EncodecConfig(target_bandwidths=[args.bandwidth], chunk_length_s = 0.2, overlap = 1. / args.fps)
audios_model = [EncodecModel.encodec_model_24khz() for id in range(args.ngpu)]
for m in audios_model:
    m.set_target_bandwidth(args.bandwidth)
    m.segment = args.chunk_length_s
    m.overlap = 1. - 1. / (args.fps * args.chunk_length_s)


# Load CLIP Model
clip_model = [torch_clip.load("ViT-B/32", device='cuda:{}'.format(id)) for id in range(args.ngpu)]

# Face Detection Model
fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                   device='cuda:{}'.format(id)) for id in range(args.ngpu)]

template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'


# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

def process_video_file(vfile, args, gpu_id):
    video_stream = cv2.VideoCapture(vfile)

    frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]

    i = -1
    for fb in batches:
        preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue

            x1, y1, x2, y2 = f
            cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])


def process_audio_file(vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    wavpath = path.join(fulldir, 'audio.wav')

    command = template.format(vfile, wavpath)
    subprocess.call(command, shell=True)

def encode_audio(vfile, args, gpu_id):
    # Load audio
    wav, sr =  torch_load(vfile)

    # Pad wav to get NoF codec
    #wav = resample(wav, orig_freq=sr, new_freq=audios_model[gpu_id].sample_rate)
    #samples_per_frame = int(0.2 * sr)
    #idx_multiplier, codes_chunks = int(1. / args.fps * sr), []

    vidname = vfile.split('/')[-2]
    dirname = vfile.split('/')[-3]
    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    #audio_chunks, i = [], 0
    #while 1:
    #    start_idx = int(i * idx_multiplier)
    #    if start_idx + samples_per_frame > len(wav[0]):
    #        break
    #    chunk = wav[:, start_idx: start_idx + samples_per_frame]
    #    audio_chunks.append(chunk)
    #    i += 1
    #batches = [audio_chunks[i:i + args.batch_size] for i in range(0, len(audio_chunks), args.batch_size)]

    #for batch in audio_chunks:
    chunk = convert_audio(wav, sr, audios_model[gpu_id].sample_rate, audios_model[gpu_id].channels)
    chunk = chunk.unsqueeze(0)
    print(chunk.shape)
    chunk = torch.nn.functional.pad(chunk, (0, 0, 0, 0, 0, args.chunk_length_s * sr), "constant", 0)
    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = audios_model[gpu_id].encode(chunk)
    #codes_chunks = torch.cat([codes for codes in encoded_frames], dim=0)
    frames = glob(path.join(fulldir, '*.jpg'))
    encoded_frames = encoded_frames[:len(frames)]
    print([encoded[0].shape for encoded in encoded_frames])
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=0)  # [B, n_q, T]
    #codes_chunks.append(np.array(codes))
    print(codes.shape)
    np.save(path.join(fulldir, 'audio_features.npy'), np.array(codes))

def encode_text(vfile, args, gpu_id):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = os.path.join(args.data_root, dirname)
    pre_fulldir = os.path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    with open(vfile, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    # Get Phones and words from json
    words = json_data['tiers']['words']['entries']
    #phones = json_data['tiers']['phones']

    frames = glob(path.join(pre_fulldir, '*.jpg'))
    m_fps = 1. / args.fps
    text_array = []
    for i in range(len(frames)):
        tmin = i * m_fps
        tmax = (i + 1) * m_fps
        tmp_word = [word for (ts, te, word) in words if ts < tmax and te >= tmin]
        text_array.append(" ".join(tmp_word))

    with torch.no_grad():
        text_tokens = torch_clip.tokenize(text_array).to(f'cuda:{gpu_id}')
        text_features = clip_model[gpu_id][0].encode_text(text_tokens)

    np.save(path.join(pre_fulldir, 'text_features.npy'), np.array(text_features.cpu().numpy()))

def mp_handler(job):
    vfile, args, gpu_id = job
    try:
        process_video_file(vfile, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()

def mp_encodec_handler(job):
    vfile, args, gpu_id = job
    try:
        encode_audio(vfile, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()

def mp_clip_hanlder(job):
    vfile, args, gpu_id = job
    try:
        encode_text(vfile, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()

def main(args):
    print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

    filelist = glob(path.join(args.data_root, '*/*.mp4'))
    # Filter list
    filelist = [vfile for vfile in filelist \
                    if not os.path.isdir(path.join(args.preprocessed_root,
                                               vfile.split('/')[-2],
                                               os.path.basename(vfile).split('.')[0]))]

    jobs = [(vfile, args, i % args.ngpu) for i, vfile in enumerate(filelist)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    print('Dumping audios...')
    for vfile in tqdm(filelist):
        try:
            process_audio_file(vfile, args)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()
            continue

    print("Extract Text Features from Clip Model")
    # Filter list
    filelist = glob((path.join(args.data_root, '*/*.json')))
    filelist = [vfile for vfile in filelist \
                    if not os.path.isfile(path.join(args.preprocessed_root,
                                               vfile.split('/')[-2], os.path.basename(vfile).split('.')[0],
                                               "text_features.npy"))]
    jobs = [(vfile, args, i % args.ngpu) for i, vfile in enumerate(filelist)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(mp_clip_hanlder, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    print("Extract Encodec Features")
    filelist = glob((path.join(args.preprocessed_root, '*/*/*.wav')))
    # Filter filelist
    filelist = [vfile for vfile in filelist \
                    if not os.path.isfile(path.join(args.preprocessed_root,
                                               vfile.split('/')[-3], vfile.split('/')[-2],
                                               "audio_features.npy"))]
    filelist = [filelist[0]]
    jobs = [(vfile, args, i % args.ngpu) for i, vfile in enumerate(filelist)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(mp_encodec_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]


if __name__ == '__main__':
    main(args)