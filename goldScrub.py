from os import path, listdir
import mxnet as mx
import cv2
from MTCNN.mtcnn_detector import MtcnnDetector
import numpy as np
import os
import time

_detector = MtcnnDetector(model_folder='MTCNN\\model', ctx=mx.cpu(0), num_worker=4, accurate_landmark=True)

def detectFaces(detector, frames, min_face_size=32):
    frames_bb = []
    fh, fw = frames[0].shape[0], frames[0].shape[1]

    for frame in frames:
        res = detector.detect_face(frame)
        if res is not None:
            bbs = []
            for r in res[0]:
                bb = [int(value) for value in r[:-1]]
                bbs.append(bb)
            frames_bb.append(bbs)
        else:
            frames_bb.append([])
    return frames_bb


def start(src, dest, test=False):
    files = os.listdir(src)
    if test:
        files = files[36:36+58]

    for file in files:
        file = os.path.join(src, file)
        img = cv2.imread(file)
        results = _detector.detect_face(img)

        if results is not None:
            boxes = results[0]
            points = results[1]

            draw = img.copy()

            for p in points:
                left_eye = p[0], p[5]
                right_eye = p[1], p[6]
                dX = right_eye[0] - left_eye[0]
                dY = right_eye[1] - left_eye[1]
                angle = np.degrees(np.arctan2(dY, dX)) - 180
                center = int(left_eye[0] + ((right_eye[0] - left_eye[0]) / 2.0)),\
                         int(left_eye[1] + ((right_eye[1] - left_eye[1]) / 2.0))


                cv2.circle(draw, (p[2],p[7]), 1, (255, 0, 0), 2)
                cv2.circle(draw, (p[3],p[8]), 1, (0, 255, 0), 2)

                cv2.circle(draw, right_eye, 1, (0, 0, 255), 2)
                cv2.circle(draw, left_eye, 1, (0, 0, 255), 2)
                cv2.circle(draw, center, 1, (255, 0, 0), 2)

                print('left eye = %s' % str(left_eye))
                print('right_eye = %s' % str(right_eye))
                print('center = %s' % str(center))
                print('angle = %s' % str(angle))

            cv2.namedWindow('detection result')
            cv2.imshow('detection result', draw)
            cv2.waitKey(0)
            cv2.destroyWindow('detection result')
