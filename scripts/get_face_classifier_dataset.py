# coding: utf-8
import os
import cv2
import numpy as np
import random
import shutil
import face_recognition
from face_detect.config import DATA_ROOT_PATH
from multiprocessing import Pool

face_classify_path = os.path.join(DATA_ROOT_PATH, 'face_classify')
face_cascade = cv2.CascadeClassifier(r'scripts/haarcascade_frontalface_default.xml')


def save_face(image_dir, save_dir):
    """
    corp face from origin image using dilb face_recognition
    :param image_dir: origin image dir
    :param save_dir: the dir of saving face
    :return:
    """
    count = 0
    for f_dir in os.listdir(image_dir):
        for f_name in os.listdir(os.path.join(image_dir, f_dir)):
            path = os.path.join(image_dir, f_dir, f_name)
            img = cv2.imread(path)
            face_locations = face_recognition.face_locations(img, model='cnn')

            for top, right, bottom, left in face_locations:
                face = img[left:right, top:bottom]
                save_path = os.path.join(save_dir, 'img-{}.jpg'.format(count))
                cols, lens, channels = face.shape
                face = cv2.resize(face, (64, int(64 / cols * cols)))
                cv2.imwrite(save_path, face)
                count += 1


def _save_face_from_vgg_dataset(anns, save_path):
    """
    1 corp face from origin image using vgg_face_dataset's annotations
    2 get non-face from origin image
    :param anns: vgg_face_dataset's annotations
    :return save_path:
    """
    pos_path = os.path.join(save_path, 'pos')
    neg_path = os.path.join(save_path, 'neg')
    if not os.path.exists(pos_path): os.makedirs(pos_path)
    if not os.path.exists(neg_path): os.makedirs(neg_path)
    pos_count = 0
    neg_count = 0
    save_size = 64

    def get_neg(image, input_size, x_1, y_1, count):
        # 以0.5的概率随机抛弃负样本
        if random.randint(0, 1) == 0:
            return count
        neg_size = save_size
        h, w, c = image.shape
        x_2, y_2 = x_1 + input_size, y_1 + input_size
        if x_1 <= 0 or x_2 > w or y_1 <= 0 or y_2 > h:
            return count
        neg = image[y_1:y_2, x_1:x_2]

        locs = face_recognition.face_locations(neg)
        if len(locs) != 0:
            return count
        print(neg.shape, os.getpid())
        neg = cv2.resize(neg, dsize=(neg_size, neg_size))
        cv2.imwrite(os.path.join(neg_path, "img-{}.jpg".format(count)), neg)
        return count + 1

    for ann in anns:
        img_path, x1, y1, x2, y2, _ = [i.strip() for i in ann.split(',')]
        x1, y1, x2, y2 = [int(i) for i in [x1, y1, x2, y2]]
        img = cv2.imread(img_path)
        if not isinstance(img, np.ndarray):
            continue
        img_h, img_w, _ = img.shape
        # pos
        dx, dy = x2 - x1, y2 - y1
        if dx > dy:
            beg_x, beg_y = x1, y1 - (dx - dy) // 2
        else:
            beg_x, beg_y = x1 - (dy - dx) // 2, y1
        max_xy = max(dx, dy)
        if not (beg_x <= 0 or beg_x + max_xy > img_w or beg_y <= 0 or beg_y + max_xy > img_h):
            face = img[beg_y: beg_y + max_xy, beg_x:beg_x + max_xy]
            face = cv2.resize(face, dsize=(save_size, save_size))
            cv2.imwrite(os.path.join(pos_path, "img-{}.jpg".format(pos_count)), face)
            pos_count += 1
        # neg
        neg_size_rand = random.randint(64, 96)
        offset = random.randint(2, 10)

        neg_count = get_neg(img, neg_size_rand, x1 - offset - neg_size_rand, y1, neg_count)
        neg_count = get_neg(img, neg_size_rand, x1, y1 - offset - neg_size_rand, neg_count)
        neg_count = get_neg(img, neg_size_rand, x2 + offset, y1, neg_count)
        neg_count = get_neg(img, neg_size_rand, x1, y2 + offset, neg_count)


# 多进程获取人脸分类的正样本与负样本
def save_face_from_vgg_dataset(ann_path, save_path, num_process=1):
    with open(ann_path, 'r') as fp:
        anns = fp.readlines()
    ann_len = len(anns)
    num_per_process = ann_len // num_process + 1
    pool_process = Pool(processes=num_process)
    start_index = 0
    for i in range(0, num_process):
        end_index = start_index + i * num_per_process
        if i == (num_process - 1):
            end_index = ann_len
        pool_process.apply_async(func=_save_face_from_vgg_dataset,
                                 args=(anns[start_index: end_index], save_path))
        start_index = end_index
    pool_process.close()
    pool_process.join()
    print('')


def split_dataset(save_path):
    os.chdir(save_path)
    train_path = os.path.join(save_path, 'train')
    dev_path = os.path.join(save_path, 'dev')
    test_path = os.path.join(save_path, 'test')

    def _move(type):
        src_p = os.path.join(save_path, type)
        imgs = os.listdir(src_p)
        length = len(imgs)
        # move train
        train_p = os.path.join(train_path, type)
        if not os.path.exists(train_p): os.makedirs(train_p)
        for i in range(0, int(length * 0.7)):
            src_path = os.path.join(src_p, imgs[i])
            shutil.move(src_path, train_p)
        # move dev
        dev_p = os.path.join(dev_path, type)
        if not os.path.exists(dev_p): os.makedirs(dev_p)
        for i in range(int(length * 0.7), int(length * 0.85)):
            src_path = os.path.join(src_p, imgs[i])
            shutil.move(src_path, dev_p)
        # move test
        test_p = os.path.join(test_path, type)
        if not os.path.exists(test_p): os.makedirs(test_p)
        for i in range(int(length * 0.85), length):
            src_path = os.path.join(src_p, imgs[i])
            shutil.move(src_path, test_p)

    _move('pos')
    _move('neg')


def detect_face(img, method='dlib'):
    # 用其他工具辅助
    if method == 'cv2':
        faces = face_cascade.detectMultiScale(
            img,
            minNeighbors=2,
            minSize=(30, 30)
        )
    elif method == 'dlib':
        faces = face_recognition.face_locations(img, model='cnn')
    else:
        raise Exception('not support this method')
    if len(faces) != 0:
        flag = True
    else:
        flag = False
    return flag


def filter_face_dataset(dataset_path, prefix):
    false_pos_path = os.path.join(face_classify_path, 'pos_false1')
    if not os.path.exists(false_pos_path):
        os.makedirs(false_pos_path)

    all_count, count = 0, 0
    for img_file in os.listdir(dataset_path):
        path = os.path.join(dataset_path, img_file)
        img = cv2.imread(path)
        if img is None or not detect_face(img):
            dst_path = os.path.join(false_pos_path, img_file)
            shutil.move(path, dst_path)
            print('{}, {}: {} move to {}'.format(all_count, count, path, false_pos_path))
            count += 1
        all_count += 1


def shuffle_img(img_dir):
    img_list = os.listdir(img_dir)
    num_img = len(img_list)
    indexs = list(range(0, num_img))
    np.random.shuffle(indexs)
    count = 0
    for img_name in img_list:
        src_path = os.path.join(img_dir, img_name)
        postfix = os.path.splitext(src_path)[-1]
        dst_path = os.path.join(img_dir, 'image-{}{}'.format(count, postfix))
        shutil.move(src_path, dst_path)
        count += 1


if __name__ == '__main__':
    # save_face_from_vgg_dataset('/data/algo/data/dataset/vgg_face/shuffle_ann.txt',
    #                           '/data/algo/data/dataset/vgg_face/face_classify')
    # split_dataset('/data/algo/data/dataset/vgg_face/face_classify')
    pass
