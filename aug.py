#-*- coding:utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import tensorflow as tf
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import utils
import time
import imgaug as ia
from imgaug import augmenters as iaa
def clahe_equalized(img):
    if len(img.shape) == 2:
        img=np.reshape(img, list(np.shape(img)) +[1])
    assert (len(img.shape)==3)  #4D arrays
    img=img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if img.shape[-1] ==3: # if color shape
        for i in range(3):
            img[:, :, i]=clahe.apply(np.array(img[:,:,i], dtype=np.uint8))
    elif img.shape[-1] ==1: # if Greys,
        img = clahe.apply(np.array(img, dtype = np.uint8))
    return img

# Rotate 90 , 180 , 270
def random_rotate_90(images):
    start_time=time.time()
    k=np.random.randint(0,4)
    images=np.rot90(images , k , axes =(1,2))
    #print 'Consume Time : ',time.time()-start_time
    return images


def tf_random_rotate_90(images):
    images = tf.py_func(random_rotate_90, [images], [tf.float64])
    return tf.convert_to_tensor(images)


# Rotate Image Manually
def random_rotate_with_PIL(image , rotate_angles = None):

    start_time=time.time()

    ### usage: map(random_rotate , images) ###
    if not np.max(image) > 1 : # if image is normalized
        image = (image * 255).astype('uint8')

    image=Image.fromarray(image)

    if rotate_angles is None:
        ind = random.randint(0, 180)
    else:
        random.shuffle(rotate_angles)
        ind = rotate_angles[0]

    minus = random.randint(0,1)
    minus=bool(minus)
    if minus==True:
        ind=ind*-1
    img = image.rotate(ind)
    img=np.asarray(img)
    consume_time = time.time() - start_time
    print consume_time
    return img

def tf_random_rotate_with_PIL(image , rotate_angles):
    image=tf.py_func(random_rotate_with_PIL , [image , rotate_angles], [tf.uint8])
    return image

def tf_aug_rotate(image , is_training , rotate_angles):
    def train(image , rotate_angles):
        image=tf_random_rotate_with_PIL(image ,rotate_angles)
        return image
    def test(image):
        return image
    image =tf.cond( is_training , lambda : train(image , rotate_angles) , lambda : test(image))
    return image

def apply_aug_rotate(images , is_training , rotate_range):
    images=tf.map_fn(lambda image : tf_aug_rotate(image , is_training , rotate_range) , images)
    return tf.convert_to_tensor(images)

#==== histogram equalization
def histo_equalized(img):
    assert (len(np.shape(img))==2)  ,' image shape : {} '.format(np.shape(img)) #4D arrays
    return cv2.equalizeHist(np.array(img, dtype = np.uint8))

def aug_lv0(image_ , is_training , crop_h , crop_w):

    def aug_with_train(image, crop_h , crop_w):
        img_h,img_w,ch=map(int , image.get_shape()[:])

        pad_w = int(img_h * 0.1)
        pad_h = int(img_w * 0.1)
        image = tf.image.resize_image_with_crop_or_pad(image, img_h+pad_h , img_w+pad_w )
        image = tf.random_crop(image, [crop_h, crop_w, ch])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        #Brightness / saturatio / constrast provides samll gains 2%~5% on cifar

        image = tf.image.random_brightness(image, max_delta=63. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.8)
        image = tf.image.per_image_standardization(image)
        return image

    def aug_with_test(image , crop_h , crop_w):

        image = tf.image.resize_image_with_crop_or_pad(image, crop_h, crop_w)
        image = tf.image.per_image_standardization(image)
        return image

    image=tf.cond(is_training , lambda : aug_with_train(image_ , crop_h, crop_w )  , \
                  lambda  : aug_with_test(image_ , crop_h, crop_w ))


    return image

def apply_aug_lv0(images, aug_fn , is_training , crop_h , crop_w  ):
    images=tf.map_fn(lambda image : aug_fn(image , is_training , crop_h , crop_w) ,  images)
    return images


def aug_lv1(images):
    seq = iaa.Sequential([
        iaa.OneOf([
            iaa.ContrastNormalization((0.5, 1.5)),
            iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
        ]),
        iaa.Affine(scale=(0.8, 1.2), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, rotate=(-30, 30)),
        iaa.OneOf([
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Dropout((0.01, 0.1), per_channel=0.5),
            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
        ]),
    ])
    augimgs = seq.augment_images(images)
    return augimgs

def aug_lv3(images):
    seq = iaa.Sequential([
        # Blur
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0, 0.5)),
            iaa.AverageBlur(k=(2, 7)),
            iaa.MedianBlur((3, 11))
        ]), iaa.OneOf([
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Dropout((0.01, 0.1), per_channel=0.5),
            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
        ]), iaa.OneOf([
            iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
            iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
            # iaa.EdgeDetect(alpha=(0.0 , 0.2))
        ]), iaa.SomeOf(2, [
            iaa.Add((-40, 40), per_channel=0.5),
            iaa.AddElementwise((-40, 40), per_channel=0.5),
            iaa.Multiply((0.5, 1.5)),
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
        ], random_order=True),
        iaa.SomeOf((0, None), [
            iaa.OneOf([
                iaa.ContrastNormalization((0.5, 1.5)),
                iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
            ]),
        ]),
        iaa.Affine(scale=(0.8, 1.2), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, rotate=(-30, 30)),
        iaa.OneOf([
            iaa.Affine(shear=(0.01, 0.05)),
            iaa.PiecewiseAffine((0.01, 0.05))
        ]),
    ], random_order=True)
    augimgs = seq.augment_images(images)
    return augimgs


if __name__ == '__main__':
    img = Image.open('/Users/seongjungkim/PycharmProjects/everyNN/3445147_20140625_L.png').resize((350, 350),
                                                                                                  Image.ANTIALIAS)
    img = np.asarray(img)
    imgs = []
    for i in range(32):
        imgs.append(img)
    augimgs=aug_lv1(imgs)
    utils.plot_images(augimgs , savepath='tmp.png')


    """
    is_training = tf.placeholder(tf.bool)
    path = 'tmp.png'
    imgs=[]
    img=np.asarray(Image.open(path)).astype('uint8')
    #img=img
    for i in range(20):
        imgs.append(np.asarray(img))
    imgs=np.asarray(imgs).astype('uint8') # numpy 로 안만들면 에러가 난다
    imgs = imgs / 255.


    #tf_imgs = apply_aug_rotate(imgs, is_training ,[90, 180, 270])
    # Rotate 90 180 270
    tf_imgs = aug_lv1(imgs)

    sess=tf.Session()
    output =sess.run(tf_imgs , feed_dict={is_training : True})
    output=np.squeeze(output)
    utils.plot_images(output , savepath = 'tmp_output.png')
"""

"""
def anchor_target_layer(rpn_cls_score, gt_boxes, im_dims, _feat_stride, anchor_scales):
    '''
    Make Python version of _anchor_target_layer_py below Tensorflow compatible
    '''
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
        tf.py_func(_anchor_target_layer_py, [rpn_cls_score, gt_boxes, im_dims, _feat_stride, anchor_scales],
                   [tf.float32, tf.float32, tf.float32, tf.float32])

    rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32), name='rpn_labels')
    rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name='rpn_bbox_targets')
    rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights, name='rpn_bbox_inside_weights')
    rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights, name='rpn_bbox_outside_weights')

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

def _anchor_target_layer_py(rpn_cls_score, gt_boxes, im_dims, _feat_stride, anchor_scales):

    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap
    im_dims = im_dims[0]
    # _anchors shape : ( 9, 4 ) anchor coordinate type : x1,y1,x2,y2

"""