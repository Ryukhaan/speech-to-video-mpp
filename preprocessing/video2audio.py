import argparse
import moviepy.editor
import os
import glob
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Directory to the dataset")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    # Check if dataset is a file or a directory
    if os.path.isfile(args.dataset):
        # Load the Video
        video = moviepy.editor.VideoFileClip("Video.mp4")
        # Extract the Audio
        audio = video.audio
        # Export the Audio
        audio.write_audiofile("A
    else:
        files = glob.glob(args.dataset + "*.mp4", recursive=True)
        for file in files:
            print(file))
            # Load the Video
            video = moviepy.editor.VideoFileClip(file)
            # Extract the Audio
            audio = video.audio
            # Export the Audio
            audio.write_audiofile(file[:-3] + "wav")
            exit()
