#-*- coding:utf-8 -*-
import random
import numpy as np
import os
import aug
import tensorflow as tf
import cifar , my_data
import sys
from PIL import Image
class Dataprovider():

    def __init__(self, datatype , batch_size ,resize , num_epoch=10 , onehot = True):
        self.resize = resize
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        n_train=None
        n_test=None
        if datatype == 'cifar_10' or datatype == 'cifar10':
            self.train_tfrecord_path = cifar.train_tfrecords # list
            self.test_tfrecord_path = cifar.test_tfrecords # list
            self.n_train =  50000 # DNN 에서 max iter 을 추정하는데 사용됩니다
            self.n_test = 10000

            self.n_classes = 10
        elif datatype == 'cifar_100' or datatype == 'cifar100':
            raise NotImplementedError
        elif datatype == 'SVNH' or datatype == 'svhn':
            raise NotImplementedError
        elif datatype == 'COCO' or datatype == 'coco':
            raise NotImplementedError
        elif datatype == 'PASCAL' or datatype == 'pascal':
            raise NotImplementedError
        elif datatype == 'MyData' or datatype == 'mydata' or datatype == 'my_data':
            self.train_tfrecord_path = my_data.train_tfrecord_path# list
            self.test_tfrecord_path = my_data.test_tfrecord_path # list
            self.n_train =  22164 # DNN 에서 max iter 을 추정하는데 사용됩니다
            self.n_test = 302
            self.n_classes = 2

        assert self.n_train is not None and self.n_test is not None , ' ** n_train : {} \t n_test : {} **'.format(self.n_train ,self.n_test)
        self.sample_image, self.sample_label, _ = self.get_sample(self.test_tfrecord_path, onehot=True,
                                                                  n_classes=self.n_classes)
        self.img_h, self.img_w, self.img_ch = np.shape(self.sample_image)
        # Resize
        if not self.resize is None:
            self.img_h, self.img_w = self.resize
        with tf.device('/cpu:0'):
            # tf.image.resize_image_with_crop_or_pad is used in 'get_shuffled_batch'
            self.batch_xs, self.batch_ys, self.batch_fs = self.get_shuffled_batch(self.train_tfrecord_path, self.batch_size,
                                                                                              self.resize, self.num_epoch)
            # Augmentation
            # self.batch_xs=self.augmentation( self.batch_xs , True , True , True)

            # One Hot
            if onehot:
                self.batch_ys = tf.one_hot(self.batch_ys, self.n_classes)
        print 'Data Infomation'
        print 'Image Height  : {} Label Width : {} Image channel : {} '.format(self.img_h , self.img_w , self.img_ch)
        print 'N classes : {}'.format(self.n_classes)
        print 'N epoch : {}'.format(self.num_epoch)

    @classmethod
    def next_batch(cls, batch_size , train_imgs , train_labs , train_fnames):
        indices = random.sample(range(np.shape(train_labs)[0]), batch_size)
        batch_xs = train_imgs[indices]
        batch_ys = train_labs[indices]
        if not train_fnames is None:
            batch_fs = train_fnames[indices]
        else:
            batch_fs = None
        return batch_xs, batch_ys, batch_fs
    @classmethod
    def cls2onehot(cls, classes, depth):
        debug_flag = False
        if not type(classes).__module__ == np.__name__:
            classes = np.asarray(classes)
            classes = classes.astype(np.int32)
        debug_flag = False
        labels = np.zeros([len(classes), depth], dtype=np.int32)
        for i, ind in enumerate(classes):
            labels[i][ind:ind + 1] = 1
        if __debug__ == debug_flag:
            print '#### data.py | cls2onehot() ####'
            print 'show sample cls and converted labels'
            print classes[:10]
            print labels[:10]
            print classes[-10:]
            print labels[-10:]
        return labels


    @classmethod
    def make_tfrecord_rawdata(cls, tfrecord_path, img_sources, labels , resize=None):
        """
        img source 에는 두가지 형태로 존재합니다 . str type 의 path 와
        numpy 형태의 list 입니다.
        :param tfrecord_path: e.g) './tmp.tfrecord'
        :param img_sources: e.g)[./pic1.png , ./pic2.png] or list flatted_imgs
        img_sources could be string , or numpy
        :param labels: 3.g) [1,1,1,1,1,0,0,0,0]
        :return:
        """
        debug_flag_lv0 = False
        debug_flag_lv1 = False
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
        img_sources_labels = zip(img_sources, labels)
        for ind, (img_source, label) in enumerate(img_sources_labels ):
            try:
                msg = '\r-Progress : {0}'.format(str(ind) + '/' + str(len(img_sources_labels )))
                sys.stdout.write(msg)
                sys.stdout.flush()
                if isinstance(img_source , str): # img source  == str
                    np_img = np.asarray(Image.open(img_source)).astype(np.int8)
                    height = np_img.shape[0]
                    width = np_img.shape[1]
                    dirpath, filename = os.path.split(img_source)
                    filename, extension = os.path.splitext(filename)
                elif type(img_source).__module__ == np.__name__: # img source  == Numpy
                    np_img = img_source
                    height , width = np.shape(img_source)[:2]
                    filename = str(ind)
                else:
                    raise AssertionError , "img_sources's element should path(str) or numpy"
                if not resize is None:
                    np_img=np.asarray(Image.fromarray(np_img).resize(resize, Image.ANTIALIAS))
                raw_img = np_img.tostring() # ** Image to String **

                if __debug__ == debug_flag_lv1:
                    print ''
                    print 'image min', np.min(np_img)
                    print 'image max', np.max(np_img)
                    print 'image shape', np.shape(np_img)
                    print 'heigth , width', height, width
                    print 'filename', filename

                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'raw_image': _bytes_feature(raw_img),
                    'label': _int64_feature(label),
                    'filename': _bytes_feature(tf.compat.as_bytes(filename))
                }))
                writer.write(example.SerializeToString())
            except IOError as ioe:
                if isinstance(img_source , str):
                    print img_source
                continue
            except TypeError as te:
                if isinstance(img_source , str):
                    print img_source
                continue
            except Exception as e:
                if isinstance(img_source , str):
                    print img_source
                print str(e)
                exit()
        writer.close()



    @classmethod
    def get_sample(cls , tfrecord_path , onehot , n_classes):
        record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)
        str_record=record_iter.next()
        example = tf.train.Example()
        example.ParseFromString(str_record)
        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
        label = int(example.features.feature['label'].int64_list.value[0])
        filename = (example.features.feature['filename'].bytes_list.value[0])
        image = np.fromstring(raw_image, dtype=np.uint8)
        image = image.reshape((height, width, -1))
        if onehot:
            label = cls.cls2onehot([label], n_classes)
        return image , label , filename

    @classmethod
    def reconstruct_tfrecord_rawdata(cls, tfrecord_path, resize):
        debug_flag_lv0 = False
        debug_flag_lv1 = False
        if __debug__ == debug_flag_lv0:
            print 'debug start | batch.py | class tfrecord_batch | reconstruct_tfrecord_rawdata '

        print 'now Reconstruct Image Data please wait a second'
        print 'Resize {}'.format(resize)
        reconstruct_image = []
        # caution record_iter is generator
        record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)

        ret_img_list = []
        ret_lab_list = []
        ret_filename_list = []
        for i, str_record in enumerate(record_iter):
            msg = '\r -progress {0}'.format(i)
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
            if not resize is None:
                image=np.asarray(Image.fromarray(image).resize(resize,Image.ANTIALIAS))
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
    @classmethod
    def get_shuffled_batch(cls , tfrecord_path, batch_size, resize , num_epoch , min_after_dequeue=500):
        resize_height, resize_width = resize
        filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=num_epoch , name='filename_queue')
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
        #image_size_const = tf.constant((resize_height, resize_width, 3), dtype=tf.int32)
        image = tf.reshape(image, image_shape)
        image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=resize_height,
                                                       target_width=resize_width)
        image=tf.cast(image , dtype=tf.float32)
        images, labels, fnames = tf.train.shuffle_batch([image, label, filename], batch_size=batch_size, capacity=10000,
                                                        num_threads=1,
                                                        min_after_dequeue=min_after_dequeue)
        return images, labels , fnames
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
        return image, label , filename
    @classmethod
    def augmentation(cls , images , randomCrop_aug , flipFlop_aug , color_aug):
        def _fn(image):
            img_h , img_w , img_ch =image.get_shape()
            img_h, img_w, img_ch=map(int , [img_h , img_w , img_ch])
            print img_h , img_w , img_ch
            if randomCrop_aug:
                image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=img_h + int(img_h * 0.1),
                                                               target_width=img_w + int(img_w * 0.1))

            if flipFlop_aug:
                image = tf.random_crop(image, [img_h , img_w , img_ch])
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_flip_up_down(image)

            #Brightness / saturatio / constrast provides samll gains 2%~5% on cifar
            if color_aug:
                image = tf.image.random_brightness(image, max_delta=63. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.8)
                image = tf.image.per_image_standardization(image)
            image = tf.minimum(image, 1.0)
            image = tf.maximum(image, 0.0)
            return image
        images = tf.map_fn(lambda image:_fn(image), images)
        return images

if '__main__' == __name__:
    Dataprovider('cifar10' , 60 , (32,32))

