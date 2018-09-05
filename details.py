""" Details
Author     : Lucas Pettit
Date       : 6/11/2018
Description: Given a folder, extract all the facial information for every image/GIF.
 The results will be saved as JSON's under dest directory with an equivalent name as the original file.
"""

from Utils import *
import mxnet as mx
from MTCNN.mtcnn_detector import MtcnnDetector
from argparse import ArgumentParser
import os
from datetime import datetime


def compare_bb(prev, curr):
    threshold = 0.2 * prev.width

    for p, c in zip(prev.toList(), curr.toList()):
        if abs(p - c) > threshold:
            return False
    return True


def detectFaces(detector, gif, min_face_size=32):
    frames_bb = []
    landmarks = []
    fh, fw, _ = gif.shape

    for frame in gif:
        res = detector.detect_face(frame)
        if res is not None:
            bbs = []
            lmks = []
            boxes = res[0]
            points = res[1]
            for b, p in zip(boxes, points):
                # prep bounding boxes (they need to be constrained t within the bounds of the image
                L, T, R, B = b[:-1]
                L = max(0, L)
                T = max(0, T)
                R = min(fw, R)
                B = min(fh, B)

                bb = Rectangle.fromLTRB([L, T, R, B])

                if bb.width >= min_face_size or bb.height >= min_face_size:
                    bb.squarify()
                    bbs.append(bb)

                    lmk = Landmarks.fromMtcnnFormat(p)
                    lmks.append(lmk)

            frames_bb.append(bbs)
            landmarks.append(lmks)

        else:
            frames_bb.append([])
            landmarks.append([])
    return frames_bb, landmarks


def trackFaces(frames_bb, frames_lmks, min_series=5):
    people = {}
    prev_bbs = []
    new_person_id = 0

    for frameId, (frame_bb, frame_lmk) in enumerate(zip(frames_bb, frames_lmks)):
        tmp_bbs = []
        for curr_bb, curr_lmk in zip(frame_bb, frame_lmk):
            # check for known issues
            if curr_bb.width != curr_bb.height:
                raise ValueError('SlitError: FrameId = %d' % frameId)
            if curr_bb.width == 0 or curr_bb.height == 0:
                raise ValueError('NoBytesError: FrameId = %d' % frameId)
            if curr_bb.width < 16 or curr_bb.height < 16:
                continue

            identified = False
            for prev, _id in prev_bbs:
                if compare_bb(prev, curr_bb):
                    people[_id]['boundingbox'].append(curr_bb)
                    people[_id]['landmark'].append(curr_lmk)
                    people[_id]['frameNum'].append(frameId)
                    tmp_bbs.append((curr_bb, _id))
                    identified = True

            if not identified:
                people[new_person_id] = {
                    'boundingbox': [curr_bb],
                    'landmark': [curr_lmk],
                    'frameNum': [frameId]
                }
                tmp_bbs.append((curr_bb, new_person_id))
                new_person_id += 1
                identified = True

        prev_bbs = list(tmp_bbs)

    # prune off people with out enough data in their series
    people = {_id: data for _id, data in people.items() if len(data['boundingbox']) >= min_series}

    frame_data = {}
    for personId, data in people.items():
        frameNums = data['frameNum']
        bbs = data['boundingbox']
        lmks = data['landmark']
        for bb, lmk, frameNum in zip(bbs, lmks, frameNums):
            if frameNum in frame_data:
                frame_data[frameNum].append({
                    'id': personId,
                    'boundingbox': bb,
                    'landmark': lmk
                })
            else:
                frame_data[frameNum] = [{
                    'id': personId,
                    'boundingbox': bb,
                    'landmark': lmk
                }]
    frame_data = [(frameNum, data) for frameNum, data in frame_data.items()]
    sorted(frame_data, key=lambda x: x[0])
    return frame_data


def MtCnnModel():
    return MtcnnDetector(model_folder='MTCNN\\model', ctx=mx.cpu(0), num_worker=4, accurate_landmark=True)


def parse_args():

    parser = ArgumentParser()
    parser.add_argument('--src', type=str, required=True,
                        help='Source directory containing images or GIFs. Can be directory tree')
    parser.add_argument('--dest', type=str, required=True,
                        help='Destination directory to save the resulting JSON files. '
                             'Destination file structure will mirror the source file structure.')
    parser.add_argument('--log-freq', type=int, default=10, help='Update log frequency')
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        raise ValueError('parse_args: \'--src\' is not a directory.')
    if os.path.isfile(args.dest):
        raise ValueError('parse_args: \'--dest\' is a file.')
    if args.log_freq < 1:
        raise  ValueError('parse_args: \'--log-freq\' must be >= 0.')

    return args


def process_files(src, dest, log_freq=10, min_series = 5):
    # make standard dirs
    dirnames = init_dirs()
    logdir = dirnames['log']
    resdir = dirnames['res']
    configdir = dirnames['config']

    script_name = os.path.basename(os.path.splitext(__file__)[0])
    logfile = os.path.join(logdir, '%s_%s_log.txt' % (script_name, os.path.basename(dest)))
    configfile = os.path.join(configdir, '%s_%s_config.json' % (script_name, os.path.basename(dest)))

    # load config file
    if os.path.isfile(configfile):
        print('loading config file')
        with open(configfile) as f:
            config = json.load(f)
    else:
        print('creating new config file')
        config = {'startIndex': 0}
        with open(configfile, 'w') as f:
            json.dump(config, f)
    start_index = config['startIndex']
    print('done')
    print('start index = %d' % start_index)

    # create log file
    if not os.path.isfile(logfile):
        open(logfile, 'w').close()

    # create dest dir
    if not os.path.isdir(dest):
        os.makedirs(dest)

    print('loading MTCNN model')
    _detector = MtCnnModel()
    print('done!')

    files = [os.path.join(src, f) for f in os.listdir(src)
             if os.path.splitext(f)[-1].lower() in ('.gif', '.png', '.jpg', '.bmp')]

    # write to log
    logstr = 'Starting silver scrub\n%d files found' % len(files)
    with open(logfile, 'a+') as f:
        f.write(logstr + '\n')
    print(logstr)

    # build tracking variables
    errors = []
    files_with_faces = 0

    # run
    for fileNum, f in enumerate(files[start_index:]):
        print('running batch %d' % fileNum)

        # dump log
        if (fileNum + 1) % log_freq == 0:
            logstr = '----------------------\n'
            logstr += 'file #       : %d-%d\n' % (fileNum + start_index - log_freq + 1, fileNum + start_index)
            logstr += 'files w/faces: %d\n' % files_with_faces
            logstr += 'fail         : %d\n' % len(errors)
            logstr += 'time         : %s\n' % datetime.now().strftime('%m/%d %H:%M:%S')

            if len(errors) > 0:
                logstr += 'Failed GIFs:\n'
                for filename, errorMessage in errors:
                    logstr += '\tfilename: %s\n\terror   : %s\n' % (filename, errorMessage)

            # print log data
            print(logstr)

            # write log file && reset vars
            with open(logfile, 'a+') as logger:
                logger.write(logstr)
            errors = []
            gif_with_face = 0

            #with open(configfile, 'w') as f:
            #   config['startIndex'] = fileNum
            #   json.dump(config, f)

        is_gif = os.path.splitext(f)[-1].lower() == '.gif'
        _min_series = min_series if is_gif else 1
        file_name = os.path.splitext(os.path.basename(f))[0]

        # scrub process starts here
        try:
            print('    loading file %s' % os.path.basename(f))
            gif = GIF.fromFile(f)

            # detect faces
            print('    detecting faces')
            bbs, lmds = detectFaces(_detector, gif)
            if len(bbs) == 0:
                continue

            files_with_faces += 1
            print('    tracking faces')
            people = trackFaces(bbs, lmds, _min_series)

            # dump people data
            if len(people) > 0:
                # convert all the int32 to int
                print('    dumping people JSON data')
                _people = []
                for frameId, p in people:
                    faces = []
                    for face in p:
                        faces.append(
                            {
                                'id': int(face['id']),
                                'boundingbox': face['boundingbox'].toList(),
                                'landmark': face['landmark'].toDict()
                            })
                    _people.append((frameId, faces))

                outfile = os.path.join(dest, '%s.json' % file_name)
                with open(outfile, 'w') as people_file:
                    print('    saving to %s' % outfile)
                    json.dump(_people, people_file)
                del _people

            del gif, people, bbs, lmds

        except Exception as e:
            errors.append((file_name, str(e)))


if __name__ == '__main__':
    # get args
    args = parse_args()

    process_files(args.src, args.dest, args.log_freq)