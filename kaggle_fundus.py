#-*- coding:utf-8 -*-
import Dataprovider
import os , glob , sys
import argparse
import tensorflow as tf 
import random 
import numpy as np 
from PIL import Image
"""bring kaggle fundus"""

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
            print n_imgs
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
    
    


def read_csv(label_path , include_first):
    f=open(label_path)
    lines=f.readlines()
    ret_paths=[]
    ret_labels=[]
    ret_dict={}

    if include_first:
        lines = lines[:]
    else:
        lines = lines[1:]

    for line in lines: # 10_right,0
        name , label = line.split(',')
        ret_dict[name] =  int(label.strip())
    return ret_dict

def get_name(path):
    names = []
    name, ext = os.path.splitext(os.path.split(path)[-1])
    return name , ext

def get_paths(imgdir , extension):
    ret_dict={}
    img_paths=glob.glob(os.path.join(imgdir , '*.'+extension))
    for path in img_paths:
        name, ext = get_name(path)
        ret_dict[name] = path
    return ret_dict

def merge_dict(dict1 , dict2):
    # dict1 , dict2 key is overlayed
    ret_dict={}
    for key in dict1:
        try:
            ret_dict[key] = [dict1[key] , dict2[key]]
        except IndexError as ie :
            print ie
            print key
            continue;
    return ret_dict

def divide_into_labels(pathLabel_dict):
    ret_dict={}
    for name in pathLabel_dict:
        path , label =pathLabel_dict[name]
        if not  label in ret_dict.keys():
            ret_dict[label] = [path]
        else:
            ret_dict[label].append(path)

    return ret_dict


def divide_into_tvt(src_dict , n_test , n_val ):
    ret_dict={}
    for label in src_dict:
        random.seed(1)
        random.shuffle(src_dict[label])

        test_paths = src_dict[label][:n_test]
        val_paths = src_dict[label][n_test:n_test + n_val ]
        train_paths = src_dict[label][n_test + n_val : ]
        ret_dict[label] = {'train' : train_paths , 'test' : test_paths , 'val' : val_paths}
    return ret_dict



def path2numpy(path , resize ):
    return np.asarray(Image.open(path).convert('RGB').resize(resize , Image.ANTIALIAS))

def paths2numpy(paths ,resize):
    imgs=[]
    for path in paths:
        try:
            imgs.append(path2numpy(path, resize))
        except IOError as ioe:
            continue;
    return imgs



if __name__ == '__main__':
    image_dir = './kaggle_fundus'
    label_path = os.path.join(image_dir , 'trainLabels.csv/trainLabels.csv')
    print 'fundus dir : {}'.format(image_dir)
    print 'label dir : {}'.format(label_path)


    labels_dict = read_csv(label_path , include_first = False )
    paths_dict  = get_paths('./kaggle_fundus/train_0', 'jpeg')
    pathLabel_dict = merge_dict(paths_dict, labels_dict)
    pathLabel_dict =divide_into_labels(pathLabel_dict)
    pathLabel_dict =divide_into_tvt(pathLabel_dict , 2,2)

    label_0_train = paths2numpy(pathLabel_dict[0]['train'],(300,300))
    label_0_test = paths2numpy(pathLabel_dict[0]['test'],(300,300))
    label_0_val = paths2numpy(pathLabel_dict[0]['val'],(300,300))

    label_1_train = paths2numpy(pathLabel_dict[1]['train'],(300,300))
    label_1_test = paths2numpy(pathLabel_dict[1]['test'],(300,300))
    label_1_val = paths2numpy(pathLabel_dict[1]['val'],(300,300))

    label_2_train = paths2numpy(pathLabel_dict[2]['train'],(300,300))
    label_2_test = paths2numpy(pathLabel_dict[2]['test'],(300,300))
    label_2_val = paths2numpy(pathLabel_dict[2]['val'],(300,300))

    label_3_train = paths2numpy(pathLabel_dict[3]['train'],(300,300))
    label_3_test = paths2numpy(pathLabel_dict[3]['test'],(300,300))
    label_3_val = paths2numpy(pathLabel_dict[3]['val'],(300,300))



    make_tfrecord('kaggle_fundus/kagglefundus_train.tfrecord', None ,(len(label_0_train), label_0_train),
                  (len(label_0_train), label_1_train), (len(label_0_train), label_2_train),
                  (len(label_0_train), label_3_train))
    make_tfrecord('kaggle_fundus/kagglefundus_val.tfrecord', None ,(len(label_0_val), label_0_val),
                  (len(label_1_val), label_1_val), (len(label_2_val), label_2_val),
                  (len(label_3_val), label_3_val))
    make_tfrecord('kaggle_fundus/kagglefundus_test.tfrecord', None, (len(label_0_test), label_0_test),
                  (len(label_1_test), label_1_test), (len(label_2_test), label_2_test),
                  (len(label_3_test), label_3_test))




    """
    paths_labels={}
    dirpath=os.path.join(image_dir , 'train_{}'.format(i) , '*.jpeg')
    label_path=os.path.join(dirpath , label_path)
    img_paths=glob.glob(dirpath)
    tfrecord_path=os.path.join(image_dir, 'fundus_{}.tfrecord'.format(i))
    #Dataprovider.Dataprovider.make_tfrecord_rawdata(tfrecord_path, img_sources=ret_paths, labels=ret_labels)
    """










