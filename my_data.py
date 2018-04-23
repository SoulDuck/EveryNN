import tensorflow as tf
import Dataprovider
import numpy as np
from PIL import Image

train_normal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/normal_train.tfrecord'
train_abnormal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/abnormal_train.tfrecord'
test_normal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/normal_test.tfrecord'
test_abnormal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/abnormal_test.tfrecord'


train_tfrecords= [train_normal_tfrecord]+[train_abnormal_tfrecord]*6
test_tfrecords = [test_abnormal_tfrecord , test_normal_tfrecord]


def get_test_imgs_labs(resize):
    normal_imgs, normal_labs, normal_fnames = Dataprovider.Dataprovider.reconstruct_tfrecord_rawdata(
        test_normal_tfrecord)
    abnormal_imgs, abnormal_labs, abnormal_fnames = Dataprovider.Dataprovider.reconstruct_tfrecord_rawdata(
        test_abnormal_tfrecord)
    img=Image.fromarray(normal_imgs[0]).resize(resize, Image.ANTIALIAS)
    imgs=[]
    for img in normal_imgs:
        img=Image.fromarray(img).resize(resize, Image.ANTIALIAS)
        imgs.append(img)
    imgs=np.asarray(imgs)
    print np.shape(imgs)
    normal_imgs=map(lambda img : Image.fromarray(img).resize(resize , Image.ANTIALIAS), normal_imgs)
    abnormal_imgs = map(lambda img: Image.fromarray(img).resize(resize, Image.ANTIALIAS), abnormal_imgs)

    test_imgs=np.vstack([normal_imgs , abnormal_imgs])
    test_labs=np.vstack([normal_labs, abnormal_labs])
    test_labs=Dataprovider.Dataprovider.cls2onehot(test_labs, 2)
    print 'Image shape : {}'.format(test_imgs)
    print 'Label shaep : {}'.format(test_labs)

    return test_imgs , test_labs

if '__main__' == __name__:
    test_imgs , test_labs=get_test_imgs_labs((300,300))
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
