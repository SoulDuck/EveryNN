import tensorflow as tf
import Dataprovider
import numpy as np
train_normal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/normal_train.tfrecord'
train_abnormal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/abnormal_train.tfrecord'
test_normal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/normal_test.tfrecord'
test_abnormal_tfrecord = './my_data/tfrecord_normal_0_10_abnormal_100_inf/abnormal_test.tfrecord'


train_tfrecords=[train_normal_tfrecord  ]
test_tfrecords = [test_normal_tfrecord , test_abnormal_tfrecord]



if '__main__' == __name__:
    #images , labels , fnames=Dataprovider.Dataprovider.reconstruct_tfrecord_rawdata(test_normal_tfrecord)
    images, labels, filenames = Dataprovider.Dataprovider.get_shuffled_batch(tfrecord_paths=[train_tfrecords],
                                                                             batch_size=10 , resize=(300,300) , num_epoch=10)
    sess=tf.Session()
    coord=tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord = coord)

    init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
    sess.run(init)

    images=sess.run(images)
    print np.shape(images)


    sess.close()
