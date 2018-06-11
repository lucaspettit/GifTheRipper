import imageio
import cv2
import os
import time
from tqdm import tqdm
from argparse import ArgumentParser


def split_duplicates(files):
    duplicates = []
    unique = {}
    failed = []

    for f in tqdm(files):
        try:
            gif = imageio.mimread(f, memtest=False)
            img = cv2.resize(gif[0], (32, 32))
            s = cv2.imencode('.jpg', img)[1].tostring()
            if s in unique:
                duplicates.append(f)
            else:
                unique[s] = f
        except Exception as e:
            failed.append(f)

    unique = list(unique.values()) + failed

    return unique, duplicates


parser = ArgumentParser()
parser.add_argument('--src', type=str, required=True, help='Folder with images')
args = parser.parse_args()

files = [os.path.join(args.src, f) for f in os.listdir(args.src)]
print('Looking for duplicates...')
uniques, duplicates = split_duplicates(files)

time.sleep(0.5)
print('%d duplicates found' % len(duplicates))
print('Deleting duplicates...')
for f in tqdm(duplicates):
    os.remove(f)

print('done')