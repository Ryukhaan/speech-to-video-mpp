import os
import glob
import sys
import argparse

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model WITH the visual quality discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)

args = parser.parse_args()

if __name__ == "__main__":
    directories = glob.glob(os.path.join(args.data_root, '*/*'))
    with open('train-lrs2.txt', 'w') as file:
        for directory in directories:
            file.write("/".join(directory.split('/')[-2:]))
            file.write('\n')
