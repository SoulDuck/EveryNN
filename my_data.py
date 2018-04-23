import tensorflow as tf
import Dataprovider
train_normal_tfrecords = './my_data/normal_train.tfrecord'
train_abnormal_tfrecords = './my_data/abnormal_train.tfrecord'
test_normal_tfrecords = './my_data/normal_test.tfrecord'
test_abnormal_tfrecords = './my_data/abnormal_test.tfrecord'


Dataprovider.read_one_example(tfrecord_path=test_normal_tfrecords , resize = (300,300))

