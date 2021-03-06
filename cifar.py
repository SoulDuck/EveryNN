from urllib import urlretrieve
import os ,sys
import zipfile
import tarfile
import glob
import numpy as np
import pickle
import Dataprovider
from PIL import Image
url = 'http://www.cs.toronto.edu/~kriz/cifar-%d-python.tar.gz' % 10
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

# Tfrecord paths
train_tfrecords = './cifar_10/cifar_10_train_imgs.tfrecord'
test_tfrecords = './cifar_10/cifar_10_test_imgs.tfrecord'
val_tfrecords = './cifar_10/cifar_10_val_imgs.tfrecord'
train_resize_54_tfrecords = './cifar_10/cifar_10_resize_54_train_imgs.tfrecord'
test_resize_54_tfrecords = './cifar_10/cifar_10_resize_54_test_imgs.tfrecord'
val_resize_54_tfrecords = './cifar_10/cifar_10_resize_54_val_imgs.tfrecord'

data_dir = './cifar_10/cifar-10-batches-py'


def report_download_progress(count , block_size , total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r {0:1%} already downloader".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def download_data_url(url, download_dir):
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir , filename)
    print 'filepath : {}'.format(file_path)
    if not os.path.exists(file_path):
        os.makedirs(download_dir)
    # Download Data
    if not os.path.exists('./cifar_10/cifar-10-python.tar.gz'):
        print "Download %s  to %s" %(url , file_path)
        file_path , _ = urlretrieve(url=url,filename=file_path,reporthook=report_download_progress)
    # Unzip Data
    print('\nExtracting files')
    if file_path.endswith(".zip"):
        zipfile.ZipFile(file=file_path , mode="r").extracall(download_dir)
    elif file_path.endswith(".tar.gz" ) or file_path.endswith(".tgz"):
        tarfile.open(name=file_path , mode='r:gz').extractall(download_dir)

def get_images_labels(*filenames):
    for i,f in enumerate(filenames):
        with open(f , mode='rb') as file:
            data = pickle.load(file)
            if i ==0:
                images=data[b'data'].reshape([-1,3,32,32])
                labels=data[b'labels']
            else:
                images=np.vstack((images,data[b'data'].reshape([-1,3,32,32])))
                labels=np.hstack((labels, data[b'labels']))

    images = images.transpose([0, 2, 3, 1])
    return images , labels


def cls2onehot(cls , depth):
    labs=np.zeros([len(cls) , depth])
    for i,c in enumerate(cls):
        labs[i,c]=1
    return labs

def get_cifar_images_labels(onehot=True , data_dir =data_dir):
    train_filenames = glob.glob(os.path.join(data_dir, 'data_batch*'))
    test_filenames = glob.glob(os.path.join(data_dir, 'test_batch*'))
    assert len(train_filenames) != 0
    assert len(test_filenames) != 0
    train_imgs, train_labs=get_images_labels(*train_filenames)
    test_imgs , test_labs=get_images_labels(*test_filenames)
    if onehot ==True:
        train_labs = cls2onehot(train_labs, 10)
        test_labs = cls2onehot(test_labs, 10)
    num_classes=10
    return train_imgs ,train_labs , test_imgs ,test_labs


# dataset name cifar_10_resize_54
def resize_32_to_54(imgs):
    ret_imgs=[]
    for img in imgs:
        ret_imgs.append(np.asarray(Image.fromarray(img).resize([54,54] , Image.ANTIALIAS)))
    return ret_imgs

if '__main__' == __name__:
    download_data_url(url , './cifar_10') # Download Dataset

    train_filenames=glob.glob(os.path.join(data_dir,'data_batch*'))
    test_filenames=glob.glob(os.path.join(data_dir, 'test_batch*'))
    test_imgs, test_labs = get_images_labels(*test_filenames)
    train_imgs , train_labs = get_images_labels(*train_filenames)

    val_imgs=train_imgs[:5000]
    val_labs = train_labs[:5000]
    train_imgs = train_imgs[5000:]
    train_labs = train_labs[5000:]
    print 'train images shape : {}'.format(np.shape(train_imgs))
    print 'Validation images shape : {}'.format(np.shape(val_imgs))
    print 'Test images shape : {}'.format(np.shape(test_imgs))

    Dataprovider.Dataprovider.make_tfrecord_rawdata(train_tfrecords, train_imgs , train_labs)
    Dataprovider.Dataprovider.make_tfrecord_rawdata(test_tfrecords, test_imgs, test_labs)
    Dataprovider.Dataprovider.make_tfrecord_rawdata(val_tfrecords, val_imgs , val_labs)

    # resize_32_to_54
    train_imgs_resized_54=resize_32_to_54(train_imgs)
    test_imgs_resized_54 = resize_32_to_54(test_imgs)
    val_imgs_resized_54 = resize_32_to_54(val_imgs)

    Dataprovider.Dataprovider.make_tfrecord_rawdata(train_resize_54_tfrecords, train_imgs_resized_54, train_labs)
    Dataprovider.Dataprovider.make_tfrecord_rawdata(test_resize_54_tfrecords, test_imgs_resized_54, test_labs)
    Dataprovider.Dataprovider.make_tfrecord_rawdata(val_resize_54_tfrecords, val_imgs_resized_54, val_labs)

    print np.shape(Dataprovider.Dataprovider.get_sample(train_tfrecords , True , 10)[0])
    print '#### Original CIFAR-10 ####'
    print 'train imgs shape : {}'.format(np.shape(train_imgs))
    print 'train labs shape : {}'.format(np.shape(train_labs))
    print 'test imgs shape : {}'.format(np.shape(test_imgs))
    print 'test labs shape : {}'.format(np.shape(test_labs))


    print np.shape(Dataprovider.Dataprovider.get_sample(train_tfrecords , True , 10)[0])
    print '#### Resized CIFAR-10 ####'
    print 'train imgs shape : {}'.format(np.shape(train_imgs_resized_54))
    print 'train labs shape : {}'.format(np.shape(train_labs))
    print 'test imgs shape : {}'.format(np.shape(test_imgs_resized_54))
    print 'test labs shape : {}'.format(np.shape(val_imgs_resized_54))



