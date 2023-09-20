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
    return ''.join(text.split(':')[1:])

def remove_footer(text):
    return ''.join(text.splitlines()[0])

if __name__ == "__main__":
    args = get_args()
    assert args.outdir[-1] == '/', print('Output directory should have / at the end')
    # Check if dataset is a file or a directory
    if os.path.isfile(args.dataset):
        text = ''
        with open(args.dataset, 'r') as file:
            text = remove_footer(remove_header(file.read()))
        with open(args.dataset, 'w') as file:
            file.write(text)
    else:
        files = glob.glob(args.dataset + "**/*.mp4", recursive=True)
        pbar = tqdm(files)
        for file in pbar:
            if os.path.isdir(file): continue
            pbar.set_description("Processing %s" % file.split('/')[-1])
            if args.outdir:
                with open(file, 'r') as f:
                    text = remove_footer(remove_header(f.read()))
                with open(args.outdir + file.split('/')[-1], 'w') as f:
                    f.write(text)
            else:
                with open(file, 'r') as f:
                    text = remove_footer(remove_header(f.read()))
                with open(file, 'w') as f:
                    f.write(text)