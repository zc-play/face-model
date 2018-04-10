# coding: utf-8
"""
生成人脸识别的数据集，
将vgg_face_dataset图像中标记人脸切分出来，按照每个人进行分类
格式如下:
    person1:
        face-1.jpg
        face-2.jpg
    person2:
        face-1.jpg
        face-2.jpg
    ...
"""
import os
import cv2
import random
import shutil
import numpy as np


def crop_face(ann_path, save_dir, is_split_dev=False):
    def _save_face(bx, by, wh, text):
        if not (bx <= 0 or bx + wh > img_w or by <= 0 or by + wh > img_h):
            face = img[by: by + wh, bx:bx + wh]
            face = cv2.resize(face, dsize=(save_size, save_size))

            file_dir = os.path.join(save_dir, img_class)
            if is_split_dev:
                if random.random() < 0.1:
                    file_dir = os.path.join(save_dir, 'dev', img_class)
                else:
                    file_dir = os.path.join(save_dir, 'train', img_class)

            os.makedirs(file_dir, exist_ok=True)
            file_path = os.path.join(file_dir, 'img-{}-{}.jpg'.format(count, text))
            cv2.imwrite(file_path, face)
            print('{}, save face: {}'.format(count, file_path))

    save_size = 64
    with open(ann_path, 'r') as fp:
        anns = fp.readlines()

    count = 0
    for ann in anns:
        img_path, x1, y1, x2, y2, _ = [i.strip() for i in ann.split(',')]
        x1, y1, x2, y2 = [int(i) for i in [x1, y1, x2, y2]]
        # check img
        img = cv2.imread(img_path)
        if not isinstance(img, np.ndarray):
            continue
        # try to make class dir
        img_class = img_path.split('/')[5]

        img_h, img_w, _ = img.shape
        # crop
        dx, dy = x2 - x1, y2 - y1
        if dx > dy:
            beg_x, beg_y = x1, y1 - (dx - dy) // 2
        else:
            beg_x, beg_y = x1 - (dy - dx) // 2, y1
        # 取比人脸更大一点的图像
        max_xy = max(dx, dy)
        increase_size = int(0.15 * max_xy)
        beg_x, beg_y = beg_x - increase_size // 2, beg_y - increase_size // 2
        dx, dy = dx + increase_size, dy + increase_size
        max_xy = max(dx, dy)
        if max_xy < 5:
            continue
        _save_face(beg_x, beg_y, max_xy, 'org')
        # augment
        if max_xy < 10:
            continue
        aug_max_xy = int((1 - random.random() * 0.3) * max_xy)
        aug_beg_x = int(beg_x + max_xy // 2 + random.randint(-30, 30) / 200 * max_xy - aug_max_xy // 2)
        aug_beg_y = int(beg_y + max_xy // 2 + random.randint(-30, 30) / 200 * max_xy - aug_max_xy // 2)
        _save_face(aug_beg_x, aug_beg_y, aug_max_xy, 'aug')
        count += 1


def split_train_dev(save_dir):
    class_list = os.listdir(save_dir)
    train_path = os.path.join(save_dir, 'train')
    dev_path = os.path.join(save_dir, 'dev')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(dev_path, exist_ok=True)
    for class_name in class_list:
        class_dir = os.path.join(save_dir, class_name)
        dev_class_dir = os.path.join(dev_path, class_name)
        os.makedirs(dev_class_dir, exist_ok=True)
        for file_name in os.listdir(class_dir):
            if random.random() < 0.1:
                shutil.move(os.path.join(class_dir, file_name), dev_class_dir)
        shutil.move(class_dir, train_path)


