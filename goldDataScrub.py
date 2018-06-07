from Utils import *
import os
import json
import cv2
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--src', type=str, required=True)
parser.add_argument('--dest', type=str, required=True)
parser.add_argument('--target-dim', type=int, default=227)
args = parser.parse_args()

if args.target_dim <= 0:
    raise ValueError('target-dim <= 0')
if not os.path.isdir(args.src):
    raise NotADirectoryError('src path (%s) is not a directory' % args.src)
if not os.path.isdir(args.dest):
    os.makedirs(args.dest)

resdir = 'res'
dataset_name = os.path.basename(args.dest)
configpath = os.path.join(resdir, '%s-config.json' % dataset_name)
target_dim = args.target_dim, args.target_dim

# load/build config object
if not os.path.isfile(configpath):
    config = {'start_index': 0}
else:
    with open(configpath) as f:
        config = json.load(f)
start_index = config['start_index']

# get list of .json files from src
files = [f for f in os.listdir(args.src) if os.path.splitext(f)[1] == '.json']


# iterate through files and perform post-processing
for i, filename in enumerate(files[start_index:]):

    # file_id
    file_id = int(os.path.splitext(filename)[0])

    # log stuff
    print('File - %d' % file_id)

    gif = GIF.fromFile('F:\\GIPHY\\Bronze\\%d.gif' % file_id)
    face_data = unpackJson(os.path.join(args.src, filename))

    people = {}

    for person_id, fds in face_data.items():
        # get nose coordinates and dimension vectors
        nose_x = [fd.landmarks.center[0] for fd in fds]
        nose_y = [fd.landmarks.center[1] for fd in fds]
        dim = [fd.boundingBox.width for fd in fds]

        # smoothify the movements
        nose_smooth_x = smooth(nose_x, 5)
        nose_smooth_y = smooth(nose_y, 5)
        dim_smooth = smooth(dim, 5)

        for fd, x, y, dim in zip(fds, nose_smooth_x, nose_smooth_y, dim_smooth):
            # update the points
            fd.centerOnPoint(x, y)
            fd.expandRegionTo(dim, dim)

            # get post-processed snip
            snip = gif.getSnip(fd.frameId, fd.boundingBox)
            snip = cv2.resize(snip, target_dim, interpolation=cv2.INTER_CUBIC)

            # write out the file
            outfilename = os.path.join(args.dest, '%d-%d-%d.jpg' % (file_id, fd.id, fd.frameId))
            cv2.imwrite(outfilename, snip)

    # update config
    config['start_index'] = i + 1
    with open(configpath, 'w') as f:
        json.dump(config, f)