import tensorflow as tf
import Dataprovider
import numpy as np
train_normal_tfrecords = './my_data/normal_train.tfrecord'
train_abnormal_tfrecords = './my_data/abnormal_train.tfrecord'
test_normal_tfrecords = './my_data/normal_test.tfrecord'
test_abnormal_tfrecords = './my_data/abnormal_test.tfrecord'


train_tfrecords=[train_normal_tfrecords  , train_abnormal_tfrecords , train_abnormal_tfrecords , train_abnormal_tfrecords]
test_tfrecords = [test_normal_tfrecords , test_abnormal_tfrecords]



if '__main__' == __name__:
    images, labels, filenames = Dataprovider.Dataprovider.get_shuffled_batch(tfrecord_paths=train_tfrecords,
                                                                             batch_size=10)

    sess=tf.Session()
    init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
    sess.run(init)
    coord=tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord = coord)

    images=sess.run(images)
    print np.shape(images)
