#-*- coding:utf-8 -*-
import tensorflow as tf
import Dataprovider
import numpy as np
from PIL import Image
import os ,sys
import random
# original Image

#항상 이런형태로 train , test tfrecords 형태로 해야한다.

train_tfrecord_path= './my_data/train.tfrecord'
test_tfrecord_path = './my_data/test.tfrecord'


def make_tfrecord(tfrecord_path, resize , normal_imgs , abnormal_imgs):
    """
    img source 에는 두가지 형태로 존재합니다 . str type 의 path 와
    numpy 형태의 list 입니다.
    :param tfrecord_path: e.g) './tmp.tfrecord'
    :param img_sources: e.g)[./pic1.png , ./pic2.png] or list flatted_imgs
    img_sources could be string , or numpy
    :param labels: 3.g) [1,1,1,1,1,0,0,0,0]
    :return:
    """
    debug_flag_lv0 = False
    debug_flag_lv1 = False
    if __debug__ == debug_flag_lv0:
        print 'debug start | batch.py | class : tfrecord_batch | make_tfrecord_rawdata'

    if os.path.exists(tfrecord_path):
        print tfrecord_path + 'is exists'
        return

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    n_abnormal = len(abnormal_imgs)
    n_normal = len(normal_imgs)
    n_train = len(normal_imgs)*2

    NORMAL =0
    ABNORMAL = 1

    total_count =0
    normal_count=0
    abnormal_count =0
    flag=True
    while(flag):
        label=random.randint(0, 1)
        if label == NORMAL and normal_count < n_normal:

            np_img=normal_imgs[normal_count]
            normal_count +=1
            ind = normal_count

        elif label == ABNORMAL and abnormal_count < n_normal:
            np_img = abnormal_imgs[abnormal_count % n_abnormal]
            abnormal_count +=1
            ind = abnormal_count

        elif normal_count + abnormal_count == n_normal*2:
            print normal_count
            print abnormal_count
            flag = False
        else:
            continue;

        height, width = np.shape(np_img)[:2]

        msg = '\r-Progress : {0}'.format(str(ind) + '/' + str(n_normal*2))
        sys.stdout.write(msg)
        sys.stdout.flush()
        if not resize is None:
            np_img = np.asarray(Image.fromarray(np_img).resize(resize, Image.ANTIALIAS))
        raw_img = np_img.tostring()  # ** Image to String **
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'raw_image': _bytes_feature(raw_img),
            'label': _int64_feature(label),
            'filename': _bytes_feature(tf.compat.as_bytes(str(ind)))
        }))
        writer.write(example.SerializeToString())
    writer.close()

if '__main__' == __name__:
    cac_dir = 'home/mediwhale-5/PythonProjects/fundus_data/cacs/imgSize_350/nor_0_10_abnor_300_inf/1/seoulfundus'
    nor_test_imgs=np.load(os.path.join(cac_dir , 'normal_test_imgs.npy'))
    abnor_test_imgs = np.load(os.path.join(cac_dir, 'abnormal_test_imgs.npy'))
    nor_train_imgs=np.load(os.path.join(cac_dir , 'normal_train_imgs.npy'))
    abnor_train_imgs = np.load(os.path.join(cac_dir, 'abnormal_train_imgs.npy'))

    make_tfrecord(train_tfrecord_path,None , nor_test_imgs , abnor_test_imgs) # Train TF Recorder
    make_tfrecord(test_tfrecord_path, None, nor_train_imgs, abnor_train_imgs) # Test TF Recorder