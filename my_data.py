import tensorflow as tf
import Dataprovider
import numpy as np
train_normal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/normal_train.tfrecord'
train_abnormal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/abnormal_train.tfrecord'
test_normal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/normal_test.tfrecord'
test_abnormal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/abnormal_test.tfrecord'


train_tfrecords= [train_normal_tfrecord]+[train_abnormal_tfrecord]*6
test_tfrecords = [test_abnormal_tfrecord , test_normal_tfrecord]


def get_test_imgs_labs():
    normal_imgs, normal_labs, normal_fnames = Dataprovider.Dataprovider.reconstruct_tfrecord_rawdata(
        test_normal_tfrecord)
    abnormal_imgs, abnormal_labs, abnormal_fnames = Dataprovider.Dataprovider.reconstruct_tfrecord_rawdata(
        test_abnormal_tfrecord)


    test_imgs=np.vstack([normal_imgs , abnormal_imgs])
    test_labs=np.vstack([normal_labs, abnormal_labs])
    return test_imgs , test_labs

if '__main__' == __name__:
    test_imgs , test_labs=get_test_imgs_labs()
    print np.shape(test_imgs)
    print np.shape(test_labs)
    images, labels, filenames = Dataprovider.Dataprovider.get_shuffled_batch(tfrecord_paths=test_tfrecords,
                                                                             batch_size=60, resize=(300, 300),
                                                                             num_epoch=120)
    sess=tf.Session()
    init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
    sess.run(init)
    coord = tf.train.Coordinator()
    for i in range(100):
        tf.train.start_queue_runners(sess=sess, coord=coord)
        imgs=sess.run(images)
        labs = sess.run(labels)
        print labs
        print np.shape(imgs)
    sess.close()
