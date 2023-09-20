import argparse
import glob
from tqdm import tqdm
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Directory to the dataset")
    parser.add_argument("--outdir", default=None, type=str, help="Directory to save audio")
    return parser.parse_args()

def remove_header(text):
    return text.split(':')[:-1]
def remove_footer(text):
    return text.splitlines()[0]

if __name__ == "__main__":
    args = get_args()
    # Check if dataset is a file or a directory
    if os.path.isfile(args.dataset):
        with open(args.dataset, 'r') as file:
            text = file.read()
            text = remove_footer(remove_header(text))
            print(text)
    else:
        files = glob.glob(args.dataset + "**/*.mp4", recursive=True)
        pbar = tqdm(files)
        for file in pbar:
            if os.path.isdir(file): continue
            pbar.set_description("Processing %s" % file.split('/')[-1])
            if args.outdir:
                with open(file, 'r') as f:
                    text = f.read()
                    text = remove_footer(remove_header(text))
                    print(text)
                #read_and_write(args.outdir + file.split('/')[-1])
            else:
                with open(file, 'r') as f:
                    text = f.read()
                    text = remove_footer(remove_header(text))
                    print(text)