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

if __name__ == "__main__":
    args = get_args()
    # Check if dataset is a file or a directory
    if os.path.isfile(args.dataset):
        # Load the Video
        video = moviepy.editor.VideoFileClip(args.dataset)
        # Extract the Audio
        audio = video.audio
        # Export the Audio
        audio.write_audiofile(args.dataset[:-3] + "wav")
    else:
        files = glob.glob(args.dataset + "*.mp4", recursive=True)
        pbar = tqdm(files)
        for file in pbar:
            pbar.set_description("Processing %s" % file.split('/')[-1])
            # Load the Video
            video = moviepy.editor.VideoFileClip(file)
            # Extract the Audio
            audio = video.audio
            # Export the Audio
            if args.outdir:
                audio.write_audiofile(args.outdir + file.split('/')[-1][:-3] + "wav", verbose=False, logger=None)
            else:
                audio.write_audiofile(file[:-3] + "wav", verbose=False, logger=None)
