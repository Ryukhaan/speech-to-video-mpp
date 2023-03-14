import argparse
import random
from functools import cache
from pathlib import Path
import os

import soundfile
import torch
import torchaudio
from einops import rearrange
from encodec import EncodecModel
from encodec.utils import convert_audio
from torch import Tensor
from tqdm import tqdm

from .config import cfg


@cache
def _load_model(device="cuda"):
    # Instantiate a pretrained EnCodec model
    assert cfg.sample_rate == 24_000
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.to(device)
    return model


def unload_model():
    return _load_model.cache_clear()


@torch.inference_mode()
def decode(codes: Tensor, device="cuda"):
    """
    Args:
        codes: (b q t)
    """
    assert codes.dim() == 3
    model = _load_model(device)
    return model.decode([(codes, None)]), model.sample_rate


def decode_to_file(resps: Tensor, path: Path):
    assert resps.dim() == 2, f"Require shape (t q), but got {resps.shape}."
    resps = rearrange(resps, "t q -> 1 q t")
    wavs, sr = decode(resps)
    soundfile.write(str(path), wavs.cpu()[0, 0], sr)


def _replace_file_extension(path, suffix):
    return (path.parent / path.name.split(".")[0]).with_suffix(suffix)


@torch.inference_mode()
def encode(wav: Tensor, sr: int, device="cuda"):
    """
    Args:
        wav: (t)
        sr: int
    """
    model = _load_model(device)
    wav = wav.unsqueeze(0)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)

    # Parameters
    n_fft = 800
    hop_size = 200
    win_size = 800
    # Padding
    #p1d = (win_size//2, win_size//2)
    #wav = torch.nn.functional.pad(wav, p1d)

    wav = wav.to(device)
    mel_chunks = []
    bar_progress = range(win_size//2, wav.shape[-1]-win_size//2+1, hop_size)
    for i in tqdm(bar_progress):
        mel_chunks.append(model.encode(wav[:, :, i-win_size//2:i+win_size//2]))
    #print(wav.shape, model.encode(wav)[0][0].shape)
    qnt = torch.cat([encoded[0][0] for encoded in mel_chunks], dim=0)  # (b q t)
    print(qnt.shape)
    return qnt


def encode_from_file(path, device="cuda"):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] == 2:
        wav = wav[:1]
    return encode(wav, sr, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--suffix", default=".wav")
    args = parser.parse_args()
    if os.path.isdir(args.folder):
        paths = [*args.folder.rglob(f"*{args.suffix}")]
        random.shuffle(paths)

        for path in tqdm(paths):
            out_path = _replace_file_extension(path, ".qnt.pt")
            if out_path.exists():
                continue
            qnt = encode_from_file(path)
            torch.save(qnt.cpu(), out_path)
    else:
        out_path = _replace_file_extension(path, ".qnt.pt")
        if out_path.exists():
            return
        qnt = encode_from_file(path)
        torch.save(qnt.cpu(), out_path)

if __name__ == "__main__":
    main()
