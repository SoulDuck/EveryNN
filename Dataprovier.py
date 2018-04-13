import random
import numpy as np
import cifar
import glob
import os
import aug
import tensorflow as tf
import cifar


class Input():
    def __init__(self, datatype):

        if datatype == 'cifar10':
            self.train_imgs, self.train_labs, self.test_imgs, self.test_labs = cifar.get_cifar_images_labels(
                onehot=True);
            self.fnames = np.asarray(range(len(self.train_labs)))

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
