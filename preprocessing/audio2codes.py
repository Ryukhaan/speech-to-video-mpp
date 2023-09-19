import os
import glob
import gc
import argparse
from encodec import EncodecModel
from encodec.utils import convert_audio

import torch
import torchaudio
import cv2
import numpy as np
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Directory to the dataset")
    parser.add_argument("--outdir", default=None, type=str, help="Directory to save audio")
    parser.add_argument("--bandwidth", default=24.0, type=float, help="Bandwidth value (by default 24.0)")
    parser.add_argument("--t", default=5, type=int, help="Number of frames as input")
    return parser.parse_args()

def read_video(video):
    video_stream = cv2.VideoCapture(video)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    number_of_frames = 0
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        number_of_frames += 1
    return number_of_frames, fps

def encode_audio(filename, model, t):
    # Load audio
    wav, sr = torchaudio.load(filename)
    # Load video to get FPS and total number of frames
    number_of_frames, fps = read_video(filename[:-3] + 'mp4')
    # Pad wav to get NoF codec
    nr = int(0.1 * sr)
    p2d = (nr, nr, 0, 0)
    wav = torch.nn.functional.pad(wav, p2d, "constant", 0)
    idx_multiplier, codes_chunks = int(1. / fps * sr), []
    # Iterate through frames
    for i, _ in enumerate(tqdm(range(number_of_frames))):
        # Get 5 previous frames
        chunk = wav[:, i * idx_multiplier: i * idx_multiplier + 2*nr]
        print(chunk.shape)
        chunk = convert_audio(chunk, sr, model.sample_rate, model.channels)
        chunk = chunk.unsqueeze(0)
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = model.encode(chunk)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
        codes_chunks.append(codes)
    assert len(codes_chunks) == number_of_frames
    # Save codec into an npy file
    np.save(filename[:-4] + '_codes.npy', np.array(codes_chunks))

if __name__ == "__main__":
    args = get_args()

    # Device Cuda or CPU and then set cache empty
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gc.collect()
    torch.cuda.empty_cache()

    # Load encodec
    audio_encodec_model = EncodecModel.encodec_model_24khz()
    audio_encodec_model.set_target_bandwidth(args.bandwidth)

    # Check if dataset is a file or a directory
    if os.path.isfile(args.dataset):
        # Load audio
        encode_audio(args.dataset, audio_encodec_model, args.t)
    else:
        files = glob.glob(args.dataset + "*.mp4", recursive=True)
        pbar = tqdm(files)
        for audio in pbar:
            pbar.set_description("Processing %s" % audio.split('/')[-1])
            name = audio.split('/')[-1]
            if args.outdir:
                encode_audio(args.outdir + audio.split('/')[-1])
            else:
                encode_audio(audio, audio_encodec_model, args.t)
