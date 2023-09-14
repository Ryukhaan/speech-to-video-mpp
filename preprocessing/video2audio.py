import argparse
import moviepy.editor
import os
import glob
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Directory to the dataset")
    parser.add_argument("--outdir", default=None, type=str, help="Directory to save audio")
    return parser.parse_args()

def read_and_write(filename):
    # Load the Video
    video = moviepy.editor.VideoFileClip(filename)
    # Extract the Audio
    audio = video.audio
    # Export the Audio (change mp4 to wav)
    audio.write_audiofile(filename[:-3] + "wav", verbose=False, logger=None)

if __name__ == "__main__":
    args = get_args()
    # Check if dataset is a file or a directory
    if os.path.isfile(args.dataset):
        read_and_write(args.dataset)
    else:
        files = glob.glob(args.dataset + "*.mp4", recursive=True)
        pbar = tqdm(files)
        for file in pbar:
            pbar.set_description("Processing %s" % file.split('/')[-1])
            if args.outdir:
                read_and_write(args.outdir + file.split('/')[-1])
            else:
                read_and_write(file)
