#-*- coding:utf-8 -*-
import tensorflow as tf
import Dataprovider
import numpy as np
from PIL import Image
import os ,sys
import random
# original Image

#항상 이런형태로 train , test tfrecords 형태로 해야한다.

def make_tfrecord(tfrecord_path, resize ,*args ):
    """
    img source 에는 두가지 형태로 존재합니다 . str type 의 path 와
    numpy 형태의 list 입니다.
    :param tfrecord_path: e.g) './tmp.tfrecord'
    :param img_sources: e.g)[./pic1.png , ./pic2.png] or list flatted_imgs
    img_sources could be string , or numpy
    :param labels: 3.g) [1,1,1,1,1,0,0,0,0]
    :return:
    """
    if os.path.exists(tfrecord_path):
        print tfrecord_path + 'is exists'
        return
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    flag=True
    n_total =0
    counts = []
    for i,arg in enumerate(args):
        print 'Label :{} , # : {} '.format(i , arg[0])
        n_total += arg[0]
        counts.append(0)

    while(flag):
        label=random.randint(0,len(args)-1)
        n_max = args[label][0]
        if counts[label] < n_max:
            imgs = args[label][1]
            n_imgs = len(args[label][1])
            ind = counts[label] % n_imgs
            np_img = imgs[ind]
            counts[label] += 1
        elif np.sum(np.asarray(counts)) ==  n_total:
            for i, count in enumerate(counts):
                print 'Label : {} , # : {} '.format(i, count )
            flag = False
        else:
            continue;

        height, width = np.shape(np_img)[:2]

        msg = '\r-Progress : {0}'.format(str(np.sum(np.asarray(counts))) + '/' + str(n_total))
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

# project 6
train_tfrecord_path = './my_data/project6/val_0_75_test_75_225/train.tfrecord'
test_tfrecord_path = './my_data/project6/val_0_75_test_75_225/test.tfrecord'
val_tfrecord_path = './my_data/project6/val_0_75_test_75_225/val.tfrecord'

# project 9
train_tfrecord_path = './my_data/project9/train.tfrecord'
test_tfrecord_path = './my_data/project9/test.tfrecord'
val_tfrecord_path = './my_data/project9/val.tfrecord'

# project 10

train_tfrecord_path = './my_data/project10/train_nor_0_abnor_1_9.tfrecord'
test_tfrecord_path = './my_data/project10/test_nor_0_abnor_1_9.tfrecord'
val_tfrecord_path = './my_data/project10/val_nor_0_abnor_1_9.tfrecord'

train_tfrecord_path = './my_data/project10/train_nor_0_abnor_10_99.tfrecord'
test_tfrecord_path = './my_data/project10/test_nor_0_abnor_10_99.tfrecord'
val_tfrecord_path = './my_data/project10/val_nor_0_abnor_10_99.tfrecord'

train_tfrecord_path = './my_data/project10/train_nor_0_10_abnor_10_inf.tfrecord'
test_tfrecord_path = './my_data/project10/test_nor_0_10_abnor_10_inf.tfrecord'
val_tfrecord_path = './my_data/project10/val_nor_0_10_abnor_10_inf.tfrecord'


train_tfrecord_path = './my_data/project10/train_nor_0_abnor_400_inf.tfrecord'
test_tfrecord_path = './my_data/project10/test_nor_0_abnor_400_inf.tfrecord'
val_tfrecord_path = './my_data/project10/val_nor_0_abnor_400_inf.tfrecord'


# Data ID 0100-0000003-015
train_tfrecord_path = './my_data/project10/train_0_9_10_55_56_inf.tfrecord'
test_tfrecord_path = './my_data/project10/test_0_9_10_55_56_inf.tfrecord'
val_tfrecord_path = './my_data/project10/val_0_9_10_55_56_inf.tfrecord'



# Data ID 0100-0000003-022
train_tfrecord_path = '../cac_regressor/0100-0000003-022/train_0_10_11_inf.tfrecord'
val_tfrecord_path = '../cac_regressor/0100-0000003-022/val_0_10_11_inf.tfrecord'
test_tfrecord_path = '../cac_regressor/0100-0000003-022/test_0_10_11_inf.tfrecord'

# Data ID 0100-0000003-020
train_tfrecord_path = '../cac_regressor/0100-0000003-020/train_0_30_31_inf.tfrecord'
val_tfrecord_path = '../cac_regressor/0100-0000003-020/val_0_30_31_inf.tfrecord'
test_tfrecord_path = '../cac_regressor/0100-0000003-020/test_0_30_31_inf.tfrecord'




# Data ID 0100-0000003-019
train_tfrecord_path = '../cac_regressor/train_0_10_11_inf.tfrecord'
val_tfrecord_path = '../cac_regressor/val_0_10_11_inf.tfrecord'
test_tfrecord_path = '../cac_regressor/test_0_10_11_inf.tfrecord'



if '__main__' == __name__:
    # project 5
    """
    cac_dir = '../fundus_data/cacs/imgSize_350/nor_0_10_abnor_300_inf/1/seoulfundus'
    nor_test_imgs=np.load(os.path.join(cac_dir , 'normal_test.npy'))
    abnor_test_imgs = np.load(os.path.join(cac_dir, 'abnormal_test.npy'))
    nor_train_imgs=np.load(os.path.join(cac_dir , 'normal_train.npy'))
    abnor_train_imgs = np.load(os.path.join(cac_dir, 'abnormal_train.npy'))
    """

    # project 6
    """
    cac_dir = '/home/mediwhale/fundus_harddisk/merged_CACS_350/1year/Numpy_Images/val_0_75_test_75_225'
    nor_test_imgs=np.load(os.path.join(cac_dir , 'normal_test.npy'))
    abnor_test_imgs = np.load(os.path.join(cac_dir, 'abnormal_test.npy'))

    nor_train_imgs=np.load(os.path.join(cac_dir , 'normal_train.npy'))
    abnor_train_imgs = np.load(os.path.join(cac_dir, 'abnormal_train.npy'))

    nor_val_imgs = np.load(os.path.join(cac_dir, 'normal_val.npy'))
    abnor_val_imgs = np.load(os.path.join(cac_dir, 'abnormal_val.npy'))

    # Train 이미지 수는 normal Image 와 똑같이 만든다
    make_tfrecord(train_tfrecord_path, None, nor_train_imgs, abnor_train_imgs , len(nor_train_imgs) , len(nor_train_imgs))
    make_tfrecord(test_tfrecord_path,None , nor_test_imgs , abnor_test_imgs , len(nor_test_imgs) , len(abnor_test_imgs)) # Train TF Recorder
    make_tfrecord(val_tfrecord_path, None, nor_val_imgs, abnor_val_imgs, len(nor_val_imgs) , len(abnor_val_imgs)) # Test TF Recorder
    """

    #project 9
    """
    cac_dir = '/home/mediwhale/fundus_harddisk/merged_reg_fundus_350'

    label_0_train=np.load(os.path.join(cac_dir , 'cac_0_train.npy'))
    label_0_val = np.load(os.path.join(cac_dir, 'cac_0_val.npy'))
    label_0_test = np.load(os.path.join(cac_dir, 'cac_0_test.npy'))

    label_1_train = np.load(os.path.join(cac_dir, 'cac_1_9_train.npy'))
    label_1_val = np.load(os.path.join(cac_dir, 'cac_1_9_val.npy'))
    label_1_test = np.load(os.path.join(cac_dir, 'cac_1_9_test.npy'))

    label_2_train = np.load(os.path.join(cac_dir, 'cac_10_99_train.npy'))
    label_2_val = np.load(os.path.join(cac_dir, 'cac_10_99_val.npy'))
    label_2_test = np.load(os.path.join(cac_dir, 'cac_10_99_test.npy'))

    label_3_train = np.load(os.path.join(cac_dir, 'cac_100_399_train.npy'))
    label_3_val = np.load(os.path.join(cac_dir, 'cac_100_399_val.npy'))
    label_3_test = np.load(os.path.join(cac_dir, 'cac_100_399_test.npy'))

    label_4_train = np.load(os.path.join(cac_dir, 'cac_400_inf_train.npy'))
    label_4_val = np.load(os.path.join(cac_dir, 'cac_400_inf_val.npy'))
    label_4_test = np.load(os.path.join(cac_dir, 'cac_400_inf_test.npy'))



    # Train 이미지 수는 normal Image 와 똑같이 만든다
    make_tfrecord(train_tfrecord_path, None, (len(label_0_train), label_0_train) ,(len(label_0_train), \
              label_1_train),(len(label_0_train), label_0_train),(len(label_0_train), label_3_train),(len(label_0_train), label_4_train))

    make_tfrecord(test_tfrecord_path,None ,(len(label_0_test), label_0_test) ,(len(label_1_test), \
              label_1_test),(len(label_2_test), label_2_test),(len(label_3_test), label_3_test),(len(label_4_test), label_4_test)) # Train TF Recorder

    make_tfrecord(val_tfrecord_path, None, (len(label_0_val), label_0_val) ,(len(label_1_val), \
              label_1_val),(len(label_2_val), label_2_val),(len(label_3_val), label_3_val),(len(label_4_val), label_4_val)) # Test TF Recorder
   

    """
    """
    #project 10 
    cac_dir = '/home/mediwhale/fundus_harddisk/merged_reg_fundus_350'

    label_0_train=np.load(os.path.join(cac_dir , 'cac_0_train.npy'))
    label_0_val = np.load(os.path.join(cac_dir, 'cac_0_val.npy'))
    label_0_test = np.load(os.path.join(cac_dir, 'cac_0_test.npy'))

    label_1_train = np.load(os.path.join(cac_dir, 'cac_400_inf_train.npy'))
    label_1_val = np.load(os.path.join(cac_dir, 'cac_400_inf_val.npy'))
    label_1_test = np.load(os.path.join(cac_dir, 'cac_400_inf_test.npy'))

    make_tfrecord(train_tfrecord_path, None, (len(label_0_train), label_0_train), (len(label_0_train), label_1_train))
    make_tfrecord(test_tfrecord_path, None, (len(label_0_test), label_0_test), (len(label_1_test), label_1_test))
    make_tfrecord(val_tfrecord_path, None, (len(label_0_val), label_0_val), (len(label_0_val), label_1_val))
    """

    # Data Id  0100-0000003-014 ==> 0100-0000003-015
    cac_dir = '/home/mediwhale/fundus_harddisk/merged_reg_fundus_350'

    label_0_train = np.load(os.path.join(cac_dir, 'cac_0_9_train.npy'))
    label_0_val = np.load(os.path.join(cac_dir, 'cac_0_9_val.npy'))
    label_0_test = np.load(os.path.join(cac_dir, 'cac_0_9_test.npy'))

    label_1_train = np.load(os.path.join(cac_dir, 'cac_10_55_train.npy'))
    label_1_val = np.load(os.path.join(cac_dir, 'cac_10_55_val.npy'))
    label_1_test = np.load(os.path.join(cac_dir, 'cac_10_55_test.npy'))

    label_2_train = np.load(os.path.join(cac_dir, 'cac_55_inf_train.npy'))
    label_2_val = np.load(os.path.join(cac_dir, 'cac_55_inf_val.npy'))
    label_2_test = np.load(os.path.join(cac_dir, 'cac_55_inf_test.npy'))


    # Train 이미지 수는 normal Image 와 똑같이 만든다
    make_tfrecord(train_tfrecord_path, None, (len(label_0_train), label_0_train), (len(label_0_train), label_1_train),
                  (len(label_0_train), label_2_train))

    make_tfrecord(test_tfrecord_path, None, (len(label_0_test), label_0_test), (len(label_1_test), label_1_test),
                  (len(label_2_test), label_2_test))  # Train TF Recorder

    make_tfrecord(val_tfrecord_path, None, (len(label_0_val), label_0_val), (len(label_1_val), label_1_val),
                  (len(label_2_val), label_2_val))  # Test TF Recorder

    ##project 10
    """
    cac_dir = '/home/mediwhale/fundus_harddisk/merged_reg_fundus_350'

    label_0_train = np.vstack([np.load(os.path.join(cac_dir, 'cac_0_train.npy')),
                              np.load(os.path.join(cac_dir, 'cac_1_9_train.npy'))])

    print np.shape(label_0_train)

    label_0_test = np.vstack([np.load(os.path.join(cac_dir, 'cac_0_test.npy')),
                              np.load(os.path.join(cac_dir, 'cac_1_9_test.npy'))])

    label_0_val = np.vstack([np.load(os.path.join(cac_dir, 'cac_0_val.npy')),
                              np.load(os.path.join(cac_dir, 'cac_1_9_val.npy'))])


    label_1_train = np.vstack([np.load(os.path.join(cac_dir, 'cac_10_99_train.npy')),
                               np.load(os.path.join(cac_dir, 'cac_100_399_train.npy')),
                               np.load(os.path.join(cac_dir, 'cac_400_inf_train.npy'))])

    label_1_test = np.vstack([np.load(os.path.join(cac_dir, 'cac_100_399_test.npy')),
                               np.load(os.path.join(cac_dir, 'cac_400_inf_test.npy'))])

    label_1_val = np.vstack([np.load(os.path.join(cac_dir, 'cac_100_399_val.npy')),
                               np.load(os.path.join(cac_dir, 'cac_400_inf_val.npy'))])


    make_tfrecord(train_tfrecord_path, None, (len(label_0_train), label_0_train), (len(label_0_train), label_1_train))
    make_tfrecord(test_tfrecord_path, None, (len(label_0_test), label_0_test), (len(label_1_test), label_1_test))
    make_tfrecord(val_tfrecord_path, None, (len(label_0_val), label_0_val), (len(label_0_val), label_1_val))
    """