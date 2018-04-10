# coding: utf-8
import os
import cv2
import time
from scripts import get_vgg_face_dataset as get_data
from scripts import get_vgg_utils as utils
from scripts import get_face_classifier_dataset as get_face
from scripts import get_recognition_face_dataset as get_rec_face
from api import face_detect, face_encoding


def test_face_detect(img_dir):
    for file_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, file_name)
        img = cv2.imread(img_path)
        t1_record = time.time()
        locations = face_detect(img, method='dlib')
        print('Dlib:  locations:{}, elapsed time: {}'.format(locations, time.time() - t1_record))

        t2_record = time.time()
        locations = face_detect(img, method='face')
        print('MyMethod: locations:{}, elapsed time: {}'.format(locations, time.time() - t2_record))


def get_data_from_url():
    """从vgg_face_dataset中的url下载图像"""
    get_data.run()


def shuffle_annotations():
    """打乱annotations"""
    utils.shuffle_annotations(utils.annotation_path, utils.shuffle_ann_path)


def get_classify_face():
    """获取人脸分类模型训练数据: 从图像中获取人脸图像, 并切分成三个数据集: train, dev, test"""
    get_face.save_face_from_vgg_dataset(utils.shuffle_ann_path, get_face.face_classify_path, num_process=4)
    get_face.split_dataset(get_face.face_classify_path)


def get_detect_face():
    """获取人脸检测模型训练数据: 从shuffle_annotations中过滤图像, 去除多人图像与无人脸图像"""
    utils.annotations_filter()


def get_recognition_face():
    anns = '/data/dataset/vgg_face/one-face-annotations.txt'
    save_dir = '/data/dataset/face-recognition'
    get_rec_face.crop_face(anns, save_dir, is_split_dev=True)


if __name__ == '__main__':
    # test_face_detect('/data/dataset/vgg_face/multi_face')
    get_recognition_face()
