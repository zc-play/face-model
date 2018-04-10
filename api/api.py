# coding: utf-8
import face_recognition
from .utils import face_detector


def face_rec():
    pass


def face_detect(img, method='face', format='cv'):
    res = []
    if method == 'dlib':
        locations = face_recognition.face_locations(img, model='hog')
        # convert (top, right, bottom, left) to (x1, y1, x2, y2)
        if format == 'cv':
            for top, right, bottom, left in locations:
                res.append([left, top, right, bottom])
    elif method == 'face':
        locations = face_detector.detect_face(img)
        if format == 'css':
            for left, top, right, bottom in locations:
                res.append([top, right, bottom, left])
    else:
        raise Exception('no support this method')
    return res


def face_encoding(face, method='face'):
    if method == 'dlib':
        encoding = face_recognition.face_encodings(face)
    elif method == 'face':
        encoding = None     # todo
    else:
        raise Exception('no support this method')
    return encoding




