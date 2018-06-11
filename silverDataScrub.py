# Silver Data Scrub
#
# 1. Queries .GIF files from <src>
# 2. Converts .GIF to set of JPG's
# 3. Saves JPG's to <dest>

import cv2
from datetime import datetime
import imageio
import os
import numpy as np
from random import randint
import json

# face detection
import mxnet as mx
from MTCNN.mtcnn_detector import MtcnnDetector


def squarify(bb):
    L, T, R, B = bb
    w, h = R - L, B - T
    size = max(w, h)
    L = int(L - ((size - w) / 2))
    T = int(T - ((size - h) / 2))
    R = int(L + size)
    B = int(T + size)
    return L, T, R, B


def resizeRect(rect, targetDim):
    if targetDim <= 0:
        raise ValueError('resizeRect: parameter "targetDim" <= 0')
    l, t, r, b = rect
    cw, ch = r - l, b - t

    dw, dh = targetDim - cw, targetDim - ch

    l = int(l - (dw / 2))
    t = int(t - (dh / 2))

    return l, t, l + targetDim, t + targetDim


def normifyBoundingBoxes(bbs):
    size = max([B - T for _, T, _, B in bbs])
    return [resizeRect(bb, size) for bb in bbs]


def MtcnnModel():
    return MtcnnDetector(model_folder='MTCNN\\model', ctx=mx.cpu(0), num_worker=4, accurate_landmark=True)


def loadFrames(paths):
    gifs = []
    for gifpath in paths:
        gif = []
        for path in gifpath:
            gif.append(cv2.imread(path))
        gifs.append(gif)
    return gifs


def markPeopleOnFrames(frames, data):
    peopleColors = {}
    for frameNum, people in data:
        frame = frames[frameNum]

        for person in people:
            if person['id'] in peopleColors:
                color = peopleColors[person['id']]
            else:
                color = randint(0, 255), randint(0, 255), randint(0, 255)
                peopleColors[person['id']] = color

            (l, t, r, b) = person['boundingbox']
            frame = cv2.rectangle(frame, (l, t), (r, b), color, 3)

    return frames


def detectFaces(detector, frames, min_face_size=32):
    frames_bb = []
    landmarks = []
    fh, fw = frames[0].shape[0], frames[0].shape[1]

    for frame in frames:
        res = detector.detect_face(frame)
        if res is not None:
            bbs = []
            lmds = []
            boxes = res[0]
            points = res[1]
            for b, p in zip(boxes, points):
                # prep bounding boxes (they need to be constrained t within the bounds of the image
                L, T, R, B = b[:-1]
                L = max(0, L)
                T = max(0, T)
                R = min(fw, R)
                B = min(fh, B)

                if (R - L) >= min_face_size and (B - T) >= min_face_size:
                    bbs.append(squarify((L, T, R, B)))

                    try:
                        right_eye = p[0], p[5]
                        left_eye = p[1], p[6]
                        center = int(right_eye[0] + ((left_eye[0] - right_eye[0]) / 2.0)), \
                                 int(right_eye[1] + ((left_eye[1] - right_eye[1]) / 2.0))
                        # load landmarks
                        lmd = {
                            'right_eye': (p[0], p[5]),
                            'left_eye': (p[1], p[6]),
                            'center_eyes': center,
                            'nose': (p[2], p[7]),
                            'right_mouth': (p[3], p[8]),
                            'left_mouth': (p[4], p[9])
                        }

                        lmds.append(lmd)
                    except Exception as e:
                        raise Exception('detectFace failed to extract landmarks: %s' % str(p))

            frames_bb.append(bbs)
            landmarks.append(lmds)

        else:
            frames_bb.append([])
            landmarks.append([])
    return frames_bb, landmarks


def playClip(frames, delay=100):
    cv2.namedWindow('display')
    for frame in frames:
        cv2.imshow('display', frame)
        cv2.waitKey(delay)
    cv2.destroyWindow('display')


def compare_bb(prev, curr):
    threshold = 0.2 * (prev[2] - prev[0])

    for p, c in zip(prev, curr):
        if abs(p - c) > threshold:
            return False
    return True


def trackFaces(frames_bb, frames_lmks, min_series=5):
    people = {}
    prev_bbs = []
    new_person_id = 0

    for frameId, (frame_bb, frame_lmk) in enumerate(zip(frames_bb, frames_lmks)):
        tmp_bbs = []
        for curr_bb, curr_lmk in zip(frame_bb, frame_lmk):
            # check for known issues
            W = curr_bb[2] - curr_bb[0]
            H = curr_bb[3] - curr_bb[1]
            if W != H:
                raise ValueError('SlitError: FrameId = %d' % frameId)
            if W == 0 or H == 0:
                raise ValueError('NoBytesError: FrameId = %d' % frameId)
            if W < 16 or H < 16:
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


def extractFrames(reader):
    # some gif's only update pixels that change. to check for this, i'm looking at the alpha channel
    # to see if there is an alpha score less than the max (255)
    hasAlphaUpdate = np.min(np.asarray(reader[1])[:, :, -1]) < 255
    frames = []

    if not hasAlphaUpdate:
        for frameId, frame in enumerate(reader):
            frame = np.asarray(frame[:, :, :3], dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    else:
        frame = np.asarray(reader[0][:, :, :3], dtype=np.uint8)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        prev = np.asarray(frame)
        for frameId, frame in zip([i + i for i in range(len(reader) - 1)], reader[1:]):
            colors = np.asarray(frame[:, :, :3], dtype=np.uint8)
            alphas = np.asarray(frame[:, :, 3], dtype=np.uint8)
            prev[alphas > 0] = 0
            curr = prev + colors
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2RGB))
            prev = curr

    return frames, hasAlphaUpdate


def cropBoundingBox(bb, frame):
    frameH, frameW = frame.shape[0], frame.shape[1]
    if len(frame.shape) == 3:
        channels = frame.shape[2]
    elif len(frame.shape) == 2:
        channels = 1
    else:
        raise ValueError('Unrecognized frame.shape: frame.shape = ' + str(frame.shape))

    left, top, right, bottom = bb
    width, height = right - left, bottom - top
    snip = np.zeros(width * height * channels, dtype=np.uint8).reshape((width, height, channels))

    capture_left = max(left, 0)
    capture_top = max(top, 0)
    capture_right = min(right, frameW)
    capture_bottom = min(bottom, frameH)

    copy_left = -left if left < 0 else 0
    copy_top = -top if top < 0 else 0
    copy_right = capture_right - capture_left + copy_left
    copy_bottom = capture_bottom - capture_top + copy_top

    snip[copy_top:copy_bottom, copy_left:copy_right] = frame[capture_top:capture_bottom, capture_left:capture_right]
    return snip


def run(src, dest):
    if not os.path.isdir(src):
        raise (NotADirectoryError('"%s" is not a directory' % src))

    #if not os.path.isdir(dest):
    #    os.makedirs(dest)

    # build detector
    print('loading MTCNN model')
    detector = MtcnnModel()
    print('done!')

    # get the list of files
    files = [os.path.join(src, f) for f in os.listdir(src) if os.path.splitext(f)[1].lower() == '.gif']
    try:
        files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except Exception as e:
        pass

    # convert and save all gifs as jpgs
    errors = []
    alphaCount = 0
    gif_with_face = 0
    logdir = 'logs'
    resdir = 'res'
    configdir = 'config'

    # build paths
    for x in (logdir, resdir, configdir):
        if not os.path.isdir(x):
            os.mkdir(x)

    logfile = os.path.join(logdir, 'silverDataScrubLog-%s.txt' % os.path.basename(dest))
    configfile = os.path.join(configdir, 'silverDataScrubConfig-%s.json' % os.path.basename(dest))


    print('loading config file')
    if os.path.isfile(configfile):
        with open(configfile) as f:
            config = json.load(f)
    else:
        config = {'startIndex': 0}
    startIndex = config['startIndex']
    print('done!')
    print('start index = %d' % startIndex)

    with open(logfile, 'a+') as f:
        f.write('Starting silver scrub\n')
        f.write('%d files found\n' % len(files))
    print('Starting silver scrub')
    print('%d files found' % len(files))

    batch_size = 10

    for fileNum, f in enumerate(files[startIndex:]):
        print('running batch %d' % fileNum)

        # dump log
        if (fileNum + 1) % batch_size == 0:
            logstr = '----------------------\n'
            logstr += 'file #      : %d-%d\n' % (fileNum + startIndex - batch_size, fileNum + startIndex + batch_size + 1)
            logstr += 'gifs w/faces: %d\n' % gif_with_face
            logstr += 'fail        : %d\n' % len(errors)
            logstr += 'num alpha   : %d\n' % alphaCount
            logstr += 'time        : %s\n' % datetime.now().strftime('%m/%d %H:%M:%S')

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
            alphaCount = 0
            logstr = ''

            # write config file
            config['startIndex'] = fileNum + startIndex
            with open(configfile, 'w') as configger:
                json.dump(config, configger)

        file_name = os.path.splitext(os.path.basename(f))[0]

        # scrub process starts here
        try:
            print('    reading GIF %s' % os.path.basename(f))
            reader = imageio.mimread(uri=f, memtest=False)

            print('    extracting frames')
            frames, hasAlphaUpdate = extractFrames(reader)
            del reader
            print('        %d frames' % len(frames))

            if hasAlphaUpdate:
                alphaCount += 1

            # detect faces
            print('    detecting faces')
            bbs, lmds = detectFaces(detector, frames)
            if len(bbs) == 0:
                continue

            gif_with_face += 1
            print('    tracking faces')
            people = trackFaces(bbs, lmds)

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
                                'boundingbox': [int(face['boundingbox'][i]) for i in range(4)],
                                'landmark': {'right_eye': [int(face['landmark']['right_eye'][i]) for i in range(2)],
                                             'left_eye': [int(face['landmark']['left_eye'][i]) for i in range(2)],
                                             'center_eyes': [int(face['landmark']['center_eyes'][i]) for i in range(2)],
                                             'right_mouth': [int(face['landmark']['right_mouth'][i]) for i in range(2)],
                                             'left_mouth': [int(face['landmark']['left_mouth'][i]) for i in range(2)],
                                             'nose': [int(face['landmark']['nose'][i]) for i in range(2)]
                                             }
                            })
                    _people.append((frameId, faces))

                with open(os.path.join(dest, '%s.json' % file_name), 'w') as people_file:
                    json.dump(_people, people_file)
                del _people

            print('    saving face snips')
            snips_saved = 0
            if(False):
                for frameId, peopleData in people:
                    for person in peopleData:
                        bb = person['boundingbox']
                        pid = person['id']

                        snip = cropBoundingBox(bb, frames[frameId])
                        cv2.imwrite(os.path.join(dest, '%d-%d-%d.jpg' % (file_name, pid, frameId)), snip)
                        snips_saved += 1

                        # explicitly delete the snip (have been having memory errors)
                        del snip
                print('        %d snips saved' % snips_saved)

            del frames, people, bbs, lmds

        except Exception as e:
            print('    Exception occured: %s' % str(e))

            errors.append((file_name, str(e)))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--src')
    parser.add_argument('--dest')
    args = parser.parse_args()
    run(args.src, args.dest)
