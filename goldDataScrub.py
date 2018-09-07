from Utils import *
import os
import json
import cv2
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--meta-src', type=str, required=True, help='Directory for .json silver data')
parser.add_argument('--data-src', type=str, required=True, help='Directory for gif files')
parser.add_argument('--dest', type=str, required=True, help='Directory to save the snips')
parser.add_argument('--resdir', type=str, required=False, default='', help='Location for resource folders')
parser.add_argument('--target-dim', type=int)
parser.add_argument('--pname', type=str, required=False, default=None, help='Name to distinguish process')
args = parser.parse_args()


if args.target_dim is not None:
    if args.target_dim <= 0:
        raise ValueError('target-dim <= 0')
if not os.path.isdir(args.meta_src):
    raise NotADirectoryError('src path (%s) is not a directory' % args.meta_src)
if not os.path.isdir(args.dest):
    os.makedirs(args.dest)

dirnames = init_dirs(rootdir=args.resdir)
resdir = dirnames['res']
configdir = dirnames['config']

# if no pname, set it to the folder name
if args.pname is None:
    args.pname = os.path.basename(args.dest)


configpath = os.path.join(configdir, 'Config-Extract-%s.json' % args.pname)
if args.target_dim is not None:
    target_dim = args.target_dim, args.target_dim
else:
    target_dim = None

# load/build config object
if not os.path.isfile(configpath):
    config = {'start_index': 0}
else:
    with open(configpath) as f:
        config = json.load(f)
start_index = config['start_index']

# get list of .json files from src

files = sorted([f for f in os.listdir(args.meta_src) if os.path.splitext(f)[-1] == '.json'],
               key=lambda x: int(os.path.splitext(x)[0]))
print('starting gold scrub')
print('%d files found' % len(files))

# iterate through files and perform post-processing
for i, filename in enumerate(files[start_index:]):

    # file_basename
    file_basename, file_ext = os.path.splitext(filename)

    # log stuff
    print('File - %s' % file_basename)

    try:

        with open(os.path.join(args.meta_src, filename)) as f:
            data = json.load(f)

        img_dim = data['shape'][:2]

        face_data = {}
        for frame_id, faces in data['frames']:
            for face in faces:
                bb = Rectangle.fromLTRB(face['boundingbox'])
                lmks = Landmarks.fromDict(face['landmark'])
                fd = FaceData(bb, lmks, face['id'], frame_id)
                if fd.id not in face_data:
                    face_data[fd.id] = [fd]
                else:
                    face_data[fd.id].append(fd)
        reader = Reader.fromFile(os.path.join(args.data_src, '%s%s' % (data['filename'], data['filetype'])))
        img_dim = data['shape'][:2]

        reader.resize(img_dim[0], img_dim[1])

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
                snip = reader.getSnip(fd.frameId, fd.boundingBox)
                if target_dim is not None:
                    snip = cv2.resize(snip, target_dim, interpolation=cv2.INTER_CUBIC)

                # write out the file
                outfilename = os.path.join(args.dest, '%s_%d_%d.jpg' % (file_basename, fd.id, fd.frameId))
                cv2.imwrite(outfilename, snip)
    except Exception as e:
        print(str(e))

    # update config
    config['start_index'] = i + 1
    with open(configpath, 'w') as f:
        json.dump(config, f)