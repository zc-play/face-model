# coding: utf-8
import os
import cv2
import random
import shutil
import numpy as np
import face_recognition
from keras_frcnn.config import DATA_ROOT_PATH


dir_path = os.path.join(DATA_ROOT_PATH, 'vgg_face')
annotation_path = os.path.join(dir_path, 'annotations.txt')
shuffle_ann_path = os.path.join(dir_path, 'shuffle_ann.txt')
part_ann_path = os.path.join(dir_path, 'part_ann.txt')


def sampling(sample_ratio):
    # 随机选取
    res = []
    with open(annotation_path, 'r') as f:
        for line in f.readlines():
            rand = random.random()
            path, left, right, top, bottom, cls = line.split(',')
            if rand > sample_ratio:
                continue
            else:
                res.append('{},{},{},{},{}, face\n'.format(path,
                                                           int(float(left)),
                                                           int(float(top)),
                                                           int(float(right)),
                                                           int(float(bottom))))
    
    with open(part_ann_path, 'w') as f:
        f.writelines(res)
        

def visualise():
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    for i in range(50):
        path, left, top, right, bottom, cls = lines[i].split(',')
        print(left, top, right, bottom)
        img = cv2.imread(path)
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255))

        cv2.imwrite('img%s.jpg' % i, img)


def shuffle_annotations(path, shuffle_ann_path):
    with open(path, 'r') as fp:
        data = fp.readlines()
    len_lines = len(data)
    indexs = list(range(0, len_lines))
    np.random.shuffle(indexs)
    res = []
    for index in indexs:
        res.append('{}\n'.format(data[index].strip()))

    with open(shuffle_ann_path, 'w', encoding='utf8') as fp:
        fp.writelines(res)


def annotations_filter(src_path=None, dst_path=None):
    """
    1 Vgg Face Dataset 数据集中一张图像对应一个人脸标记，在frcnn学习中，会把标记之外的，划分为负样本(bg);
    因此得去掉存在多个人脸的图像
    2 部分图像存在标记错误
    策略： 先用dlib库处理，再人工标注筛选
    :param src_path:
    :param dst_path:
    """
    root_path = os.path.join(DATA_ROOT_PATH, 'vgg_face')
    if not src_path:
        src_path = os.path.join(root_path, 'annotations.txt')
    if not dst_path:
        dst_path = os.path.join(root_path, 'one-face-annotations.txt')
    no_face_ann_path = os.path.join(root_path, 'no-face-annotations.txt')
    multi_face_ann_path = os.path.join(root_path, 'multi-face-annotations.txt')

    no_face_dir = os.path.join(root_path, 'no_face')
    multi_face_dir = os.path.join(root_path, 'multi_face')
    os.makedirs(no_face_dir, exist_ok=True)
    os.makedirs(multi_face_dir, exist_ok=True)
    no_face_cnt, multi_face_cnt, count = 0, 0, 0

    with open(src_path, 'r') as src_fp:
        src_anns = src_fp.readlines()
    dst_fp = open(dst_path, 'w')
    no_face_fp = open(no_face_ann_path, 'w')
    multi_face_fp = open(multi_face_ann_path, 'w')

    try:
        for ann in src_anns:
            path = ann.split(',')[0]
            img = cv2.imread(path)
            if img is None:
                if os.path.exists(path):
                    os.remove(path)
                continue
            locations = face_recognition.face_locations(img)
            length = len(locations)
            postfix = os.path.splitext(path)[-1]
            if length == 0:
                # 图像复制到文件夹下, 人工筛选
                no_face_path = os.path.join(no_face_dir, 'img-nf-{}{}'.format(no_face_cnt, postfix))
                shutil.move(path, no_face_path)
                print('{}, no_face, {}, {} move to {} '.format(count, no_face_cnt, path, no_face_path))
                no_face_cnt += 1
                no_face_fp.write(ann)
            elif length > 1:
                # 图像复制到文件夹下, 人工筛选
                multi_face_path = os.path.join(multi_face_dir, 'img-mf-{}{}'.format(multi_face_cnt, postfix))
                shutil.move(path, multi_face_path)
                print('{}, multi_face, {}, {} move to {} '.format(count, multi_face_cnt, path, multi_face_path))
                multi_face_cnt += 1
                multi_face_fp.write(ann)
            else:
                dst_fp.write(ann)
                dst_fp.flush()
            count += 1
    finally:
        dst_fp.close()
        no_face_fp.close()
        multi_face_fp.close()


if __name__ == '__main__':
    # visualise()
    # sampling()
    shuffle_annotations(annotation_path, shuffle_ann_path)




