from urllib import urlretrieve
import os ,sys
import zipfile
import tarfile
import glob
import numpy as np
import pickle
import Dataprovider
url = 'http://www.cs.toronto.edu/~kriz/cifar-%d-python.tar.gz' % 10
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

train_tfrecords = ['./cifar_10/cifar_10_train_imgs.tfrecord']
test_tfrecords = ['./cifar_10/cifar_10_test_imgs.tfrecord']
data_dir = './cifar_10/cifar-10-batches-py'
def report_download_progress(count , block_size , total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r {0:1%} already downloader".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def download_data_url(url, download_dir):
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir , filename)
    if not os.path.exists(file_path):
        try:
            os.makedirs(download_dir)
        except Exception :
            pass

        print "Download %s  to %s" %(url , file_path)
        file_path , _ = urlretrieve(url=url,filename=file_path,reporthook=report_download_progress)
        print file_path
        print('\nExtracting files')
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path , mode="r").extracall(download_dir)
        elif file_path.endswith(".tar.gz" , ".tgz"):
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





if '__main__' == __name__:
    print isinstance('str' , str)
    print isinstance(np.array([1,2,3]), str)
    a=np.array([0,0,0])
    print type(a).__module__ == np.__name__
    #download_data_url(url , './cifar_10') # Download Dataset
    train_filenames=glob.glob(os.path.join(data_dir,'data_batch*'))
    test_filenames=glob.glob(os.path.join(data_dir, 'test_batch*'))
    train_imgs , train_labs = get_images_labels(*train_filenames)
    test_imgs, test_labs = get_images_labels(*test_filenames)
    Dataprovider.Dataprovider.make_tfrecord_rawdata( train_tfrecords, train_imgs , train_labs)
    Dataprovider.Dataprovider.make_tfrecord_rawdata(test_tfrecords, test_imgs, test_labs)
    print np.shape(Dataprovider.Dataprovider.get_sample(train_tfrecords , True , 10)[0])
    print 'train imgs shape : {}'.format(np.shape(train_imgs))
    print 'train labs shape : {}'.format(np.shape(train_labs))
    print 'test imgs shape : {}'.format(np.shape(test_imgs))
    print 'test labs shape : {}'.format(np.shape(test_labs))



