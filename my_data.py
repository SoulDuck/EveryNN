#-*- coding:utf-8 -*-
import tensorflow as tf
import Dataprovider
import numpy as np
from PIL import Image
import os
# original Image
"""
train_normal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/normal_train.tfrecord'
train_abnormal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/abnormal_train.tfrecord'
test_normal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/normal_test.tfrecord'
test_abnormal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/abnormal_test.tfrecord'
"""
# Resize fundus 350 x 350
train_normal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/350_350/normal_train.tfrecord'
train_abnormal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/350_350/abnormal_train.tfrecord'
test_normal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/350_350/normal_test.tfrecord'
test_abnormal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/350_350/abnormal_test.tfrecord'

train_tfrecords= [train_normal_tfrecord]+[train_abnormal_tfrecord]*6
test_tfrecords = [test_abnormal_tfrecord , test_normal_tfrecord]

def resize_train_test_imgs(resize , save_folder):
    #원본 이미지를 바로 resize 하면 잘 학습이 너무 느리다.
    #그래서 이미지를 줄이고 그 이미지를 다시 tfrecord로 만드는 코드이다.

    tfrecord_paths = [test_normal_tfrecord, test_abnormal_tfrecord,test_normal_tfrecord, test_abnormal_tfrecord]
    tfrecord_paths = [train_normal_tfrecord , train_abnormal_tfrecord , test_normal_tfrecord , test_abnormal_tfrecord]
    normal_train, abnormal_train, normal_test, abnormal_test = map(
        lambda path: Dataprovider.Dataprovider.reconstruct_tfrecord_rawdata(path, resize), tfrecord_paths)
    print ''
    print 'normal Train Image shape : {}'.format(np.shape(np.asarray(normal_train[1])))
    print 'normal Train label :', str(normal_train[1][:10])+'...'
    print 'abnormal Train label :', str(abnormal_train[1][:10])+'...'
    print 'normal Test label :', str(normal_test[1][:10])+'...'
    print 'abnormal Test label :', str(abnormal_test[1][:10])+'...'
    tfrecord_paths=map(lambda path : os.path.join(save_folder , os.path.split(path)[-1]) ,  tfrecord_paths)
    Dataprovider.Dataprovider.make_tfrecord_rawdata(tfrecord_paths[0] , normal_train[0], normal_train[1])
    Dataprovider.Dataprovider.make_tfrecord_rawdata(tfrecord_paths[1], abnormal_train[0], abnormal_train[1])
    Dataprovider.Dataprovider.make_tfrecord_rawdata(tfrecord_paths[2], normal_test[0], normal_test[1])
    Dataprovider.Dataprovider.make_tfrecord_rawdata(tfrecord_paths[3], abnormal_test[0], abnormal_test[1])

def get_test_imgs_labs(resize):
    test_labs=[]
    normal_imgs, normal_labs, normal_fnames = Dataprovider.Dataprovider.reconstruct_tfrecord_rawdata(
        test_normal_tfrecord , None)
    abnormal_imgs, abnormal_labs, abnormal_fnames = Dataprovider.Dataprovider.reconstruct_tfrecord_rawdata(
        test_abnormal_tfrecord , None)
    normal_imgs=map(lambda img : np.asarray(Image.fromarray(img).resize(resize , Image.ANTIALIAS)), normal_imgs)
    abnormal_imgs = map(lambda img: np.asarray(Image.fromarray(img).resize(resize, Image.ANTIALIAS)), abnormal_imgs)

    test_imgs=np.vstack([normal_imgs , abnormal_imgs])
    test_labs.extend(normal_labs )
    test_labs.extend(abnormal_labs)
    print test_labs
    test_labs=Dataprovider.Dataprovider.cls2onehot(test_labs, 2)
    print 'Image shape : {}'.format(np.shape(test_imgs))
    print 'Label shape : {}'.format(np.shape(test_labs))

    return test_imgs/255. , test_labs

if '__main__' == __name__:

    img , lab , fname =Dataprovider.Dataprovider.get_sample(train_normal_tfrecord , None , n_classes=2)
    print np.shape(img)
    print lab
    print fname

    img , lab , fname =Dataprovider.Dataprovider.get_sample(test_normal_tfrecord , None , n_classes=2)
    print np.shape(img)
    print lab
    print fname

    normal_imgs, normal_labs, normal_fnames = Dataprovider.Dataprovider.reconstruct_tfrecord_rawdata(
        test_normal_tfrecord, None)
    print normal_labs

    img, lab, fname = Dataprovider.Dataprovider.reconstruct_tfrecord_rawdata(test_normal_tfrecord)
    print np.shape(img)
    print lab
    print fname

    img, lab, fname = Dataprovider.Dataprovider.get_sample(train_abnormal_tfrecord, None, n_classes=2)
    print np.shape(img)
    print lab
    print fname


    #원본 이미지를 줄인다.
    test_imgs, test_labs = get_test_imgs_labs((350, 350))
    exit()
    resize_train_test_imgs((350,350), './my_data/tfrecord_normal_0_10_abnormal_100_inf/350_350' )
    #test_imgs , test_labs=get_test_imgs_labs((300,300))
    print np.shape(np.asarray(test_imgs))
    print np.shape(test_labs)
    images, labels, filenames = Dataprovider.Dataprovider.get_shuffled_batch(tfrecord_paths=test_tfrecords,
                                                                             batch_size=60, resize=(300, 300),
                                                                             num_epoch=120)
    sess=tf.Session()
    init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
    sess.run(init)
    coord = tf.train.Coordinator()
    for i in range(10):
        tf.train.start_queue_runners(sess=sess, coord=coord)
        imgs=sess.run(images)
        labs = sess.run(labels)
        print labs
        print np.shape(imgs)
    sess.close()
