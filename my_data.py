import tensorflow as tf
import Dataprovider
import numpy as np
train_normal_tfrecords = './my_data/normal_train.tfrecord'
train_abnormal_tfrecords = './my_data/abnormal_train.tfrecord'
test_normal_tfrecords = './my_data/normal_test.tfrecord'
test_abnormal_tfrecords = './my_data/abnormal_test.tfrecord'





def read_one_example( tfrecord_path, resize):
    filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=10)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       # Defaults are not specified since both keys are required.
                                       features={
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'raw_image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'filename': tf.FixedLenFeature([], tf.string)

                                       })
    image = tf.decode_raw(features['raw_image'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    filename = tf.cast(features['filename'], tf.string)

    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)
    if not resize == None:
        resize_height, resize_width = resize
        image_size_const = tf.constant((resize_height, resize_width, 3), dtype=tf.int32)
        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                       target_height=resize_height,
                                                       target_width=resize_width)
    return image, label, filename
image, label ,filename =read_one_example(tfrecord_path=test_normal_tfrecords , resize = (300,300))
print np.shape(image)
print label
print filename

sess=tf.Session()
init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init)
print np.shape(sess.run(image))