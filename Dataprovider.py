import random
import numpy as np
import cifar
import glob
import os
import aug
import tensorflow as tf
import cifar
import sys

class Input():
    def __init__(self, datatype):
        if datatype == 'cifar_10' or 'cifar10':
            self.train_imgs, self.train_labs, self.test_imgs, self.test_labs = cifar.get_cifar_images_labels(
                onehot=True);
            self.fnames = np.asarray(range(len(self.train_labs)))

        elif datatype == 'cifar_100' or 'cifar100':
            raise NotImplementedError

        elif datatype == 'SVNH' or 'svhn':
            raise NotImplementedError

        elif datatype == 'COCO' or 'coco':
            raise NotImplementedError

        elif datatype == 'PASCAL' or 'pascal':
            raise NotImplementedError


    def next_batch(self, batch_size):
        indices = random.sample(range(np.shape(self.train_labs)[0]), batch_size)
        batch_xs = self.train_imgs[indices]
        batch_ys = self.train_labs[indices]
        if not self.fnames is None:
            batch_fs = self.fnames[indices]
        else:
            batch_fs = None
        return batch_xs, batch_ys, batch_fs

    def cls2onehot(self, cls, depth):
        debug_flag = False
        if not type(cls).__module__ == np.__name__:
            cls = np.asarray(cls)
        cls = cls.astype(np.int32)
        debug_flag = False
        labels = np.zeros([len(cls), depth], dtype=np.int32)
        for i, ind in enumerate(cls):
            labels[i][ind:ind + 1] = 1
        if __debug__ == debug_flag:
            print '#### data.py | cls2onehot() ####'
            print 'show sample cls and converted labels'
            print cls[:10]
            print labels[:10]
            print cls[-10:]
            print labels[-10:]
        return labels

    def return_traindata(self):
        return self.train_imgs, self.train_labs

    def return_testdata(self):
        return self.test_imgs, self.test_labs

    def aug(aug_flag, random_crop_resize, x_, is_training):
        if aug_flag:
            print 'aug : True'
            if random_crop_resize is None:
                random_crop_resize = int(x_.get_shape()[-2])

            x_ = tf.map_fn(lambda image: aug.aug_lv0(image, is_training, image_size=random_crop_resize), x_)
            x_ = tf.identity(x_, name='aug_')

        return x_

    @classmethod
    def make_tfrecord_rawdata(cls, tfrecord_path, paths, labels):
        """
        :param tfrecord_path: e.g) './tmp.tfrecord'
        :param paths: e.g)[./pic1.png , ./pic2.png]
        :param labels: 3.g) [1,1,1,1,1,0,0,0,0]
        :return:
        """
        debug_flag_lv0 = True
        debug_flag_lv1 = True
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
        paths_labels = zip(paths, labels)
        error_file_paths = []
        for ind, (path, label) in enumerate(paths_labels):
            try:
                msg = '\r-Progress : {0}'.format(str(ind) + '/' + str(len(paths_labels)))
                sys.stdout.write(msg)
                sys.stdout.flush()

                np_img = np.asarray(Image.open(path)).astype(np.int8)
                height = np_img.shape[0]
                width = np_img.shape[1]
                raw_img = np_img.tostring()
                dirpath, filename = os.path.split(path)
                filename, extension = os.path.splitext(filename)
                if __debug__ == debug_flag_lv1:
                    print ''
                    print 'image min', np.min(np_img)
                    print 'image max', np.max(np_img)
                    print 'image shape', np.shape(np_img)
                    print 'heigth , width', height, width
                    print 'filename', filename
                    print 'extension ,', extension

                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'raw_image': _bytes_feature(raw_img),
                    'label': _int64_feature(label),
                    'filename': _bytes_feature(tf.compat.as_bytes(filename))
                }))
                writer.write(example.SerializeToString())
            except IndexError as ie:
                print path
                continue
            except IOError as ioe:
                print path
                continue
            except Exception as e:
                print path
                print str(e)
                continue
        writer.close()

    @classmethod
    def get_iterator(tfrecord_path):
        record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)

    @classmethod
    def reconstruct_tfrecord_rawdata(tfrecord_path):
        debug_flag_lv0 = True
        debug_flag_lv1 = True
        if __debug__ == debug_flag_lv0:
            print 'debug start | batch.py | class tfrecord_batch | reconstruct_tfrecord_rawdata '

        print 'now Reconstruct Image Data please wait a second'

        reconstruct_image = []
        # caution record_iter is generator

        record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)
        n = len(list(record_iter))
        record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)

        print 'The Number of Data :', n
        ret_img_list = []
        ret_lab_list = []
        ret_filename_list = []
        for i, str_record in enumerate(record_iter):
            msg = '\r -progress {0}/{1}'.format(i, n)
            sys.stdout.write(msg)
            sys.stdout.flush()

            example = tf.train.Example()
            example.ParseFromString(str_record)

            height = int(example.features.feature['height'].int64_list.value[0])
            width = int(example.features.feature['width'].int64_list.value[0])
            raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
            label = int(example.features.feature['label'].int64_list.value[0])
            filename = (example.features.feature['filename'].bytes_list.value[0])
            image = np.fromstring(raw_image, dtype=np.uint8)
            image = image.reshape((height, width, -1))
            ret_img_list.append(image)
            ret_lab_list.append(label)
            ret_filename_list.append(filename)
        ret_img = np.asarray(ret_img_list)
        ret_lab = np.asarray(ret_lab_list)
        if debug_flag_lv1 == True:
            print ''
            print 'images shape : ', np.shape(ret_img)
            print 'labels shape : ', np.shape(ret_lab)
            print 'length of filenames : ', len(ret_filename_list)
        return ret_img, ret_lab, ret_filename_list

    def get_shuffled_batch(cls , tfrecord_path, batch_size, resize):
        resize_height, resize_width = resize
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

        image_shape = tf.stack([height, width, 3])  # image_shape shape is ..
        image_size_const = tf.constant((resize_height, resize_width, 3), dtype=tf.int32)
        image = tf.reshape(image, image_shape)
        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                       target_height=resize_height,
                                                       target_width=resize_width)
        images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=30, num_threads=1,
                                                min_after_dequeue=10)
        return images, labels
    @classmethod
    def read_one_example(cls , tfrecord_path, resize):
        filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=10)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           # Defaults are not specified since both keys are required.
                                           features={
                                               'height': tf.FixedLenFeature([], tf.int64),
                                               'width': tf.FixedLenFeature([], tf.int64),
                                               'raw_image': tf.FixedLenFeature([], tf.string),
                                               'label': tf.FixedLenFeature([], tf.int64)
                                           })
        image = tf.decode_raw(features['raw_image'], tf.uint8)
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        label = tf.cast(features['label'], tf.int32)
        image_shape = tf.pack([height, width, 3])
        image = tf.reshape(image, image_shape)
        if not resize == None:
            resize_height, resize_width = resize
            image_size_const = tf.constant((resize_height, resize_width, 3), dtype=tf.int32)
            image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                           target_height=resize_height,
                                                           target_width=resize_width)
        return image, label