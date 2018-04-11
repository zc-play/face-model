# coding: utf-8
import io
import os
import imghdr
import gevent
from urllib import request
from gevent.queue import Queue
from gevent import monkey
from face_detect.config import DATA_ROOT_PATH

monkey.patch_all()


dir_path = os.path.join(DATA_ROOT_PATH, 'vgg_face')
VGG_FILE = os.path.join(dir_path, 'files')
VGG_FACE_IMAGE = os.path.join(dir_path, 'images')
VGG_FACE_ANNOTATION = os.path.join(dir_path, 'annotations.txt')
if not os.path.exists(VGG_FACE_IMAGE):
    os.makedirs(VGG_FACE_IMAGE)

q = Queue(200)
q_save = Queue(100)  # 同步保存annotations
timeout = 20


def parse_file():
    file_list = os.listdir(VGG_FILE)
    for f_name in file_list:
        f_path = os.path.join(VGG_FILE, f_name)
        f_name = os.path.splitext(f_name)[0]

        with open(f_path, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                try:
                    line = [i.strip() for i in line.split(' ')]
                    if int(line[-1]) == 0 and float(line[-2]) >= 2:
                        continue
                    img_id, url, left, top, right, bottom = line[0:6]
                    q.put([f_name, img_id, url, left, top, right, bottom])
                    count += 1
                except:
                    pass
                if count >= 50:
                    break
    print(u'文件已全部读取')


def get_img_from_url():
    while True:
        if q.empty():
            gevent.sleep(5)
            if q.empty():
                break

        f_name, img_id, url, left, top, right, bottom = q.get()
        try:
            req = request.urlopen(url, timeout=timeout)
            data = req.read()
            # 校验图片格式
            img_type = imghdr.what(io.BytesIO(data))
            if img_type not in ['jpeg', 'png', 'bmp']:
                print('file format error:' + url)
                continue
            img_type = 'jpg' if img_type == 'jpeg' else img_type
            name_path = os.path.join(VGG_FACE_IMAGE, f_name)
            if not os.path.exists(name_path):
                os.mkdir(name_path)
            path = os.path.join(name_path, '{}-{}.{}'.format(f_name, img_id, img_type))
            with open(path, 'wb') as f:
                f.write(data)
                print('save succeed:' + path)
            ann = '{},{},{},{},{},face\n'.format(path, int(float(left)),
                                                 int(float(top)),
                                                 int(float(right)),
                                                 int(float(bottom)))
            q_save.put(ann)
        except Exception as e:
            print(e)
            print('save failed:' + url)


def save_annotations():
    f = open(VGG_FACE_ANNOTATION, 'w')
    while True:
        print("q.size:{}, q_save:{}".format(q.qsize(), q_save.qsize()))
        if q_save.empty() and q.empty():
            gevent.sleep(timeout * 1.5)
            if q_save.empty() and q.empty():
                break
        ann = q_save.get()
        f.write(ann)
        f.flush()
    f.close()


def run():
    threads = [gevent.spawn(parse_file)]
    threads += [gevent.spawn(func) for func in [get_img_from_url] * 100]
    threads += [gevent.spawn(save_annotations)]

    gevent.joinall(threads)


if __name__ == '__main__':
    run()
