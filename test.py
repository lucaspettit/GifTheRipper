import numpy as np
import imageio
import cv2

def cropBoundingBox(bb, frame):
    frameH, frameW = frame.shape[0], frame.shape[1]
    if len(frame.shape) == 3:
        channels = frame.shape[2]
    elif len(frame.shape) == 2:
        channels = 1
    else:
        raise ValueError('Unrecognized frame.shape: frame.shape = ' + str(frame.shape))

    if i == 3:
        print('OK!')

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




gif = imageio.mimread('res\\28.gif')
frames, _ = extractFrames(gif)
people = [(0, [{'id': 0, 'boundingbox': (143, 24, 310, 191)}]), (1, [{'id': 0, 'boundingbox': (145, 24, 312, 191)}]), (2, [{'id': 0, 'boundingbox': (142, 16, 309, 183)}]), (3, [{'id': 0, 'boundingbox': (134, -1, 301, 166)}]), (4, [{'id': 0, 'boundingbox': (133, -3, 300, 164)}]), (5, [{'id': 0, 'boundingbox': (128, -2, 295, 165)}]), (6, [{'id': 0, 'boundingbox': (120, -14, 287, 153)}]), (7, [{'id': 0, 'boundingbox': (119, -13, 286, 154)}]), (8, [{'id': 0, 'boundingbox': (119, -12, 286, 155)}]), (9, [{'id': 0, 'boundingbox': (112, -12, 279, 155)}]), (10, [{'id': 0, 'boundingbox': (112, -12, 279, 155)}]), (12, [{'id': 1, 'boundingbox': (116, -6, 275, 153)}]), (13, [{'id': 1, 'boundingbox': (116, -7, 275, 152)}]), (14, [{'id': 1, 'boundingbox': (117, -4, 276, 155)}]), (15, [{'id': 1, 'boundingbox': (121, -1, 280, 158)}]), (16, [{'id': 1, 'boundingbox': (117, 0, 276, 159)}]), (17, [{'id': 1, 'boundingbox': (117, -1, 276, 158)}]), (18, [{'id': 1, 'boundingbox': (122, -2, 281, 157)}]), (19, [{'id': 1, 'boundingbox': (123, -5, 282, 154)}]), (20, [{'id': 1, 'boundingbox': (132, -2, 291, 157)}]), (21, [{'id': 1, 'boundingbox': (127, 2, 286, 161)}]), (22, [{'id': 1, 'boundingbox': (129, 1, 288, 160)}]), (23, [{'id': 1, 'boundingbox': (133, 10, 292, 169)}]), (24, [{'id': 1, 'boundingbox': (133, 7, 292, 166)}]), (25, [{'id': 1, 'boundingbox': (139, 2, 298, 161)}]), (26, [{'id': 1, 'boundingbox': (133, 2, 292, 161)}]), (27, [{'id': 1, 'boundingbox': (144, 10, 303, 169)}]), (28, [{'id': 1, 'boundingbox': (153, 8, 312, 167)}]), (29, [{'id': 1, 'boundingbox': (154, 9, 313, 168)}]), (30, [{'id': 1, 'boundingbox': (165, 13, 324, 172)}]), (31, [{'id': 1, 'boundingbox': (166, 13, 325, 172)}])]

snips = []

for i, (p, frame) in enumerate(zip(people, frames)):
    print(i)
    bb = p[1][0]['boundingbox']
    snip = cropBoundingBox(bb, frame)
    snips.append(snip)