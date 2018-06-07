import numpy as np
import json
import imageio
import cv2
import os


class Rectangle(object):

    # INIT METHODS
    ###########################################################
    def init(self):
        self._top = 0
        self._left = 0
        self._bottom = 0
        self._right = 0
        self._rtype = None

    def _initializer_(self, left, top, right, bottom, rtype=int):
        # check tha the numbers are correct order
        if (right <= left) or (bottom <= top):
            raise ValueError('Rectangle: Invalid dementions. (right <= left) or (bottom <- top)')
        if rtype not in (int, float):
            raise TypeError('Rectangle.init: invalid rtype %s' % type(rtype))

        self._rtype = rtype
        self._left = rtype(left)
        self._right = rtype(right)
        self._top = rtype(top)
        self._bottom = rtype(bottom)
        self._width = rtype(right - left)
        self._height = rtype(bottom - top)

        return self

    @classmethod
    def fromLTRB(cls, values, rtype=int):
        if len(values) != 4:
            raise ValueError('Rectangle.fromLTRB: values does not contain 4 elements')
        new_class = cls()
        return new_class._initializer_(values[0], values[1], values[2], values[3], rtype=rtype)

    @classmethod
    def fromXYWH(cls, values, rtype=int):
        if len(values) != 4:
            raise ValueError('Rectangle.fromXYWH: values does not contain 4 elements')
        new_class = cls()
        return new_class._initializer_(values[0], values[1], values[0] + values[2], values[1] + values[3],
                                       rtype=rtype)

    @classmethod
    def fromJSON(cls, json_str, encoding='lrtb'):
        values = json.loads(json_str)
        new_class = cls()

        if encoding == 'lrtb':
            return cls.fromLTRB(values)
        elif encoding == 'xywh':
            return cls.fromXYWH(values)

    # GETTERS
    ###########################################################
    @property
    def top(self):
        return self._top

    @property
    def bottom(self):
        return self._bottom

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def x(self):
        return self._left

    @property
    def y(self):
        return self._top

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def center(self):
        return self.x + (self.width / 2.0), self.y + (self.height / 2.0)

    # SETTERS
    ###########################################################
    @top.setter
    def top(self, value):
        if value < self._bottom:
            value = self._rtype(value)
            self._top = value
            self._height = self._bottom - self._top
        else:
            raise ValueError('Rectangle.top [setter]: top cannot be >= bottom')

    @bottom.setter
    def bottom(self, value):
        if value > self._top:
            value = self._rtype(value)
            self._bottom = value
            self._height = self._bottom - self._top
        else:
            raise ValueError('Rectangle.bottom [setter]: bottom cannot be <= top')

    @left.setter
    def left(self, value):
        if value < self._right:
            if self._rtype == int:
                value = self._rtype(round(value))
            else:
                value = self._rtype(value)
            self._left = value
            self._width = self._right - self._left
        else:
            raise ValueError('Rectangle.left [setter]: left cannot be >= right')

    @right.setter
    def right(self, value):
        if value > self._left:
            value = self._rtype(value)
            self._right = value
            self._width = self._right - self._left
        else:
            raise ValueError('Rectangle.right [setter]: right cannot be <= left')

    @x.setter
    def x(self, value):
        if value < self._right:
            value = self._rtype(value)
            self._left = value
            self._width = self._right - self._left
        else:
            raise ValueError('Rectangle.x [setter]: x cannot be >= right')

    @y.setter
    def y(self, value):
        if value < self._bottom:
            value = self._rtype(value)
            self._top = value
            self.height = self._bottom - self._top
        else:
            raise ValueError('Rectangle.y [setter]: y cannot be >= bottom')

    @width.setter
    def width(self, value):
        if value > 0:
            value = self._rtype(value)
            self._right = self._left + value
            self._width = value
        else:
            raise ValueError('Rectangle.width [setter]: width cannot be <= 0')

    @height.setter
    def height(self, value):
        if value > 0:
            value = self._rtype(value)
            self._bottom = self._top + value
            self._height = value
        else:
            raise ValueError('Rectangle.height [setter]: height cannot be <= 0')

    # PUPLIC METHODS
    ###########################################################

    def toJSON(self, encoding='ltrb', **kwargs):
        if encoding == 'ltrb':
            json.dumps([self.left, self.top, self.right, self.bottom])
        elif encoding == 'xywh':
            json.dumps([self.x, self.y, self.width, self.height])
        else:
            raise ValueError('Rectangle.toJSON: Invalid encoding value %s' % encoding)

    def moveByVector(self, vector: np.ndarray):
        dX = vector[0]
        dY = vector[1]
        self.move(dX, dY)

    def move(self, dX, dY):
        self._left = self._rtype(self._left + dX)
        self._right = self._rtype(self._left + self._width)
        self._top = self._rtype(self._top + dY)
        self._bottom = self._rtype(self._top + self._height)

    def squarify(self):
        size = max(self.width, self.height)
        self.left = self.left - ((size - self.width) / 2.0)
        self.top = self.top - ((size - self.height) / 2.0)
        self.right = self.left + size
        self.bottom = self.top + size

    def resize(self, dW, dH):
        if dW + self.width <= 0 or dH + self.height <= 0:
            raise ValueError('Rectangle.resize: delta parameter too large. Will create negative/zero dimension')

        self.left = self.left - (dW / 2.0)
        self.top = self.top - (dH / 2.0)
        self.right = self.right + (dW / 2.0)
        self.bottom = self.bottom + (dH / 2.0)

    def resizeTo(self, targetW, targetH):
        if targetW <= 0 or targetH <= 0:
            raise ValueError('Rectangle.resizeTo: parameter targetW and targetH must be > 0')
        dW = targetW - self.width
        dH = targetH - self.height
        self.resize(dW, dH)

    def resizeByRatio(self, ratio):
        if ratio < 0:
            raise ValueError('Rectangle.resizeByRatio: ratio parameter must be > 0')
        dW = (ratio * self.width) - self.width
        dH = (ratio * self.height) - self.height
        self.resize(dW, dH)

    # Calls or whatever
    ###########################################################
    def __str__(self):
        s = 'x: %s, y: %s, width: %s, height: %s'
        s %= (str(self.x), str(self.y), str(self.width), str(self.height))
        return s


class Landmarks(object):
    def __init__(self):
        self._left_eye = np.zeros(2)
        self._right_eye = np.zeros(2)
        self._center_eyes = np.zeros(2)
        self._left_mouth = np.zeros(2)
        self._right_mouth = np.zeros(2)
        self._face_center = np.zeros(2)
        self._nose = np.zeros(2)
        self._spread = 0.0
        self._ltype = float

        # landmark point getter dictionary
        self._getter_dict = {
            'right_eye': lambda: self.right_eye,
            'left_eye': lambda: self.left_eye,
            'center_eyes': lambda: self.center_eyes,
            'nose': lambda: self.nose,
            'left_mouth': lambda: self.left_mouth,
            'right_mouth': lambda: self.right_mouth,
            'center': lambda: self.center

        }

    def __str__(self):
        s = 'right_eye: %s, left_eye: %s, center_eyes: %s, nose: %s, right_mouth: %s, left_mouth: %s'
        s %= (str(self.right_eye), str(self._left_eye), str(self.center_eyes),
              str(self.nose), str(self.right_mouth), str(self.left_mouth))
        return s

    @classmethod
    def fromDict(cls, landmarkDict, ltype=float):
        if ltype not in (float, int):
            raise ValueError('Landmarks._initializer_: Invalid ltype %s' % type(ltype))

        new_class = cls()

        new_class._right_eye = np.array([landmarkDict['right_eye'][0],
                                         landmarkDict['right_eye'][1]],
                                        dtype=np.float32)
        new_class._left_eye = np.array([landmarkDict['left_eye'][0],
                                        landmarkDict['left_eye'][1]],
                                       dtype=np.float32)
        new_class._center_eyes = np.array([landmarkDict['center_eyes'][0],
                                           landmarkDict['center_eyes'][1]],
                                          dtype=np.float32)
        new_class._nose = np.array([landmarkDict['nose'][0],
                                    landmarkDict['nose'][1]],
                                   dtype=np.float32)
        new_class._right_mouth = np.array([landmarkDict['right_mouth'][0],
                                           landmarkDict['right_mouth'][1]],
                                          dtype=np.float32)
        new_class._left_mouth = np.array([landmarkDict['left_mouth'][0],
                                          landmarkDict['left_mouth'][1]],
                                         dtype=np.float32)

        points = (new_class._right_eye, new_class.left_eye, new_class.nose, new_class.right_mouth, new_class.left_mouth)
        center_x = np.mean(np.array([point[0] for point in points]))
        center_y = np.mean(np.array([point[1] for point in points]))
        new_class._face_center = np.array([center_x, center_y])

        distance = np.mean(np.array([
            np.linalg.norm(new_class._face_center - point)
            for point in (new_class._right_eye, new_class._left_eye, new_class._right_mouth, new_class._left_mouth)
        ]))

        new_class._spread = distance

        return new_class

    @staticmethod
    def fromMtcnnFormat(points, ltype=float):
        right_eye = points[0], points[5]
        left_eye = points[1], points[6]
        nose = points[2], points[7]
        right_mouth = points[3], points[8]
        left_mouth = points[4], points[9]
        center_eyes = int(right_eye[0] + ((left_eye[0] - right_eye[0]) / 2.0)), \
                      int(right_eye[1] + ((left_eye[1] - right_eye[1]) / 2.0))

        return Landmarks.fromDict({
            'right_eye': right_eye,
            'left_eye': left_eye,
            'nose': nose,
            'right_mouth': right_mouth,
            'left_mouth': left_mouth,
            'center_eyes': center_eyes
        }, ltype=ltype)

    @staticmethod
    def fromJSON(json_str, ltype=float):
        lmkDict = json.loads(json_str)
        return Landmarks.fromDict(lmkDict)

    def _toLtype(self, point):
        return self._ltype(point[0]), self._ltype(point[1])

    @property
    def right_eye(self):
        return self._toLtype(self._right_eye)

    @property
    def left_eye(self):
        return self._toLtype(self._left_eye)

    @property
    def center_eyes(self):
        return self._toLtype(self._center_eyes)

    @property
    def nose(self):
        return self._toLtype(self._nose)

    @property
    def right_mouth(self):
        return self._toLtype(self._right_mouth)

    @property
    def left_mouth(self):
        return self._toLtype(self._left_mouth)

    @property
    def center(self):
        return self._toLtype(self._face_center)

    @property
    def point_names(self):
        return list(self._getter_dict.keys())

    @property
    def var(self):
        return np.var(np.array([self._right_eye, self._left_eye, self._nose, self._right_mouth, self._left_mouth]))

    @property
    def std(self):
        return np.std(np.array([self._right_eye, self._left_eye, self._nose, self._right_mouth, self._left_mouth]))

    @property
    def mean(self):
        return np.mean(np.array([self._right_eye, self._left_eye, self._nose, self._right_mouth, self._left_mouth]))

    @property
    def max(self):
        return np.max([np.array([
            np.linalg.norm(self._nose - point) for point in
            (self._right_eye, self._left_eye, self._right_mouth, self._left_mouth)
        ])])

    def getPointByName(self, point_name: str):
        if point_name not in self._getter_dict:
            raise ValueError('Landmarks.getPoint: %s is not a point name' % point_name)
        return self._getter_dict[point_name]()

    def toDict(self):
        d = {'right_eye': self.right_eye,
             'left_eye': self.left_eye,
             'center_eyes': self.center_eyes,
             'nose': self.nose,
             'right_mouth': self.right_mouth,
             'left_mouth': self.left_mouth}
        return d

    def toJSON(self):
        return json.dumps(self.toDict())

    def resizeByRatio(self, ratio, origin=None):
        # TODO: figure this one out :'(
        pass

    def resizeTo(self, width, height, origin=None):
        # TODO: figure this one out too :'(
        pass


class FaceData(object):
    def __init__(self, boundingBox: Rectangle, landmarks: Landmarks, faceId, frameId):
        self._rect = boundingBox
        self._landmarks = landmarks
        self._id = faceId
        self._frame_id = frameId

    @property
    def boundingBox(self):
        return self._rect

    @property
    def landmarks(self):
        return self._landmarks

    @property
    def id(self):
        return self._id

    @property
    def frameId(self):
        return self._frame_id

    def centerOnLandmarkPoint(self, point_name: str):
        x, y = self.landmarks.getPointByName(point_name)
        self.centerOnPoint(x, y)

    def centerOnPoint(self, x, y):
        _x, _y = self._rect.center
        dX = x - _x
        dY = y - _y
        self._rect.move(dX, dY)

    def expandRegionByRatio(self, ratio, origin=None):
        self._rect.resizeByRatio(ratio)

    def expandRegionTo(self, width, height, origin=None):
        self._rect.resizeTo(targetW=width, targetH=height)


class GIF(object):
    def __init__(self):
        self._frames = []
        self._num_frames = 0
        self._width = 0
        self._height = 0
        self._encoding = 'standard'  # standard, alpha
        self._current_frame = 0
        self._channels = 0

    @classmethod
    def fromFile(cls, path: str):
        reader = imageio.mimread(uri=path, memtest=False)
        new_class = cls()
        new_class._frames, alpha = GIF._extractFrames(reader)
        new_class._num_frames = len(new_class._frames)
        new_class._encoding = 'alpha' if alpha else 'standard'

        # get dims
        frame = new_class._frames[0]
        new_class._height = frame.shape[0]
        new_class._width = frame.shape[1]

        # get channels
        if len(frame.shape) == 3:
            new_class._channels = frame.shape[2]
        elif len(frame.shape) == 2:
            new_class._channels = 1
        else:
            raise Exception('GIF.fromFile: Unable to identify channels: GIF.shape = %s' % str(frame.shape))

        return new_class

    @classmethod
    def fromFrames(cls, frames: [np.ndarray], encoding='standard'):
        if len(frames) == 0:
            raise ValueError('GIF.fromFrames: Unable to create GIF. Not enough frames provided.')

        frameSize = frames[0].shape
        if len(frameSize) not in (3, 2):
            raise ValueError('GIF.fromFrames: Unable to create GIF. Unable to identify channels: GIF.shape = %s' % str(frameSize))
        for i, frame in enumerate(frames[1:]):
            if len(frame.shape) != len(frameSize):
                raise  ValueError('GIF.fromFrames: Unable to create GIF. Frame dimensions mismatched on frame number %d.' % i)
            for a, b in zip(frameSize, frame.shape):
                if a != b:
                    raise ValueError('GIF.fromFrames: Unable to create GIF. Frame size mismatched on frame number %d. %s != %s'
                                     % (i, str(frameSize), str(frame.shape)))

        if encoding not in ('alpha', 'standard'):
            raise ValueError('GIF.fromFrames: Unable to create GIF. Encoding value "%s" is invalid.' % encoding)

        new_class = cls()
        new_class._frames = frames
        new_class._num_frames = len(frames)
        new_class._encoding = encoding
        new_class._height = frameSize[0]
        new_class._width = frameSize[1]

        if len(frameSize) == 3:
            new_class._channels = frameSize[2]
        else:
            new_class._channels = 1

        return new_class

    @classmethod
    def fromFiles(cls, paths: [str]):
        frames = []
        for p in paths:
            frame = cv2.imread(p)
            frames.append(frame)
        return cls.fromFrames(frames)

    @staticmethod
    def _extractFrames(reader):
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

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_frame >= self._num_frames:
            self._current_frame = 0
            raise StopIteration
        else:
            self._current_frame += 1
            return self._frames[self._current_frame - 1]

    def __getitem__(self, index):
        return self._frames[index]

    def __len__(self):
        return self._num_frames

    def __str__(self):
        s = 'GIF (num. frames: %d, shape: %s)' % (self._num_frames, str([int(x) for x in self.shape]))
        return s

    @property
    def encoding(self):
        return self._encoding

    @property
    def shape(self):
        return self._height, self._width, self._channels

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def channels(self):
        return self._channels

    def getSnip(self, index, rect: Rectangle):
        frame = self[index]

        snip = np.zeros(rect.width * rect.height * self.channels, dtype=np.uint8)
        snip = snip.reshape((rect.height, rect.width, self.channels))

        capture_left = max(rect.left, 0)
        capture_top = max(rect.top, 0)
        capture_right = min(rect.right, self.width)
        capture_bottom = min(rect.bottom, self.height)

        copy_left = -rect.left if rect.left < 0 else 0
        copy_top = -rect.top if rect.top < 0 else 0
        copy_right = capture_right - capture_left + copy_left
        copy_bottom = capture_bottom - capture_top + copy_top

        snip[copy_top:copy_bottom, copy_left:copy_right] = frame[capture_top:capture_bottom, capture_left:capture_right]
        return snip

    def save(self, path, as_images=False, **kwargs):
        if not isinstance(path, str):
            raise ValueError('GIF.save: Invalid parameter type. Argument \'path\' must be type str.')

        if as_images:
            root, ext = os.path.splitext(path)

            for i, frame in enumerate(self._frames):
                p = root + ('-%d%s' % (i, ext))
                cv2.imwrite(p, frame)
        else:
            writer = imageio.get_writer(path, **kwargs)
            try:
                for frame in self._frames:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.append_data(frame)
            finally:
                writer.close()


class Player(object):

    @staticmethod
    def play(frames, delay=40, displayName='display'):
        cv2.namedWindow(displayName)
        for frame in frames:
            cv2.imshow(displayName, frame)
            cv2.waitKey(delay)
        cv2.destroyWindow(displayName)


def unpackJson(path):
    face_data = {}
    with open(path) as f:
        facial_detect_data = json.load(f)
    for frame_id, faces in facial_detect_data:
        for face in faces:
            bb = Rectangle.fromLTRB(face['boundingbox'])
            lmks = Landmarks.fromDict(face['landmark'])
            fd = FaceData(bb, lmks, face['id'], frame_id)
            if fd.id not in face_data:
                face_data[fd.id] = [fd]
            else:
                face_data[fd.id].append(fd)
    return face_data


def smooth(y, window_size=5):
    window = np.ones(window_size) / window_size
    pad_size = int(window_size / 2)
    _y = [y[0]] * pad_size + y + [y[-1]] * pad_size
    y_smooth = np.convolve(_y, window, mode='same')
    return y_smooth[pad_size:-pad_size]


if __name__ == '__main__':

    from argparse import ArgumentParser
    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser.add_argument('--gif_src', type=str, required=True)
    parser.add_argument('--json_src', type=str, required=True)
    parser.add_argument('--delay', type=int, default=80)
    parser.add_argument('--dest', type=str, required=False, default=None)
    args = parser.parse_args()

    delay = args.delay
    saveFile = args.dest is not None
    if not os.path.isdir(args.dest):
        os.makedirs(args.dest)

    # get gif name
    gifname = os.path.basename(os.path.splitext(args.gif_src)[0])

    # load gif
    gif = GIF.fromFile(args.gif_src)

    # build a FaceData object for every face in each frame
    face_data = unpackJson(args.json_src)
    target_dim = (256, 256)
    raw_snips = {}
    snips = {}

    for i, (person_id, fds) in enumerate(face_data.items()):
        nose_x = [fd.landmarks.center[0] for fd in fds]
        nose_y = [fd.landmarks.center[1] for fd in fds]
        dim = [fd.boundingBox.width for fd in fds]

        nose_smooth_x = smooth(nose_x, 5)
        nose_smooth_y = smooth(nose_y, 5)
        dim_smooth = smooth(list(smooth(dim, 10)), 5)
        framenums = [i for i in range(len(nose_x))]

        # make graph of movements
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.scatter(framenums, nose_x, s=4, alpha=0.5, color='b')
        ax2.scatter(framenums, nose_y, s=4, alpha=0.5, color='g')
        ax3.scatter(framenums, dim, s=4, alpha=0.5, color='c')
        ax1.plot(framenums, nose_smooth_x, color='k')
        ax2.plot(framenums, nose_smooth_y, color='k')
        ax3.plot(framenums, dim_smooth, color='k')
        ax1.set_title('X Position')
        ax2.set_title('Y Position')
        ax3.set_title('Dimension')
        ax3.set_xlabel('Frame Number')
        ax1.set_ylabel('Value')
        ax2.set_ylabel('Value')
        ax3.set_ylabel('Value')
        fig.tight_layout()
        plt.savefig(os.path.join(args.dest, '%s-%d-smoothing-graph.png' % (gifname, i)), bbox_inches='tight')

        for fd, x, y, dim in zip(fds, nose_smooth_x, nose_smooth_y, dim_smooth):
            # get raw snip
            snip = gif.getSnip(fd.frameId, fd.boundingBox)
            snip = cv2.resize(snip, target_dim, interpolation=cv2.INTER_CUBIC)
            if fd.id not in raw_snips:
                raw_snips[fd.id] = [snip]
            else:
                raw_snips[fd.id].append(snip)

            fd.centerOnPoint(x, y)
            fd.expandRegionTo(dim, dim)

            # get post-processed snip
            snip = gif.getSnip(fd.frameId, fd.boundingBox)
            snip = cv2.resize(snip, target_dim, interpolation=cv2.INTER_CUBIC)
            if fd.id not in snips:
                snips[fd.id] = [snip]
            else:
                snips[fd.id].append(snip)

    for i in snips.keys():
        frames = []
        for raw, processed in zip(raw_snips[i], snips[i]):
            frames.append(np.hstack((raw, processed)))
        #Player.play(frames, displayName='person %d' % i, delay=delay)

        newGif = GIF.fromFrames(raw_snips[i])
        newGif.save(os.path.join(args.dest, '%s-%d-raw.gif' % (gifname, i)), as_images=False)

        newGif = GIF.fromFrames(snips[i])
        newGif.save(os.path.join(args.dest, '%s-%d-smoothed.gif' % (gifname, i)), as_images=False)

        newGif = GIF.fromFrames(frames)
        newGif.save(os.path.join(args.dest, '%s-%d-comparison.gif' % (gifname, i)), as_images=False)

    print('ok')
