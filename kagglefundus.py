#-*- coding:utf-8 -*-


import Dataprovider
import os , glob
import sys
import argparse
"""bring kaggle fundus"""
parser=argparse.ArgumentParser()
parser.add_argument('--datadir' , type= str , default='./kaggle_fundus')
args=parser.parse_args()

datadir=args.datadir
print 'fundus dir : {}'.format(datadir)
labelpath = 'trainLabels.csv/trainLabels.csv'
for i in range(5):
    paths_labels={}
    dirpath=os.path.join(datadir , 'train_{}'.format(i) , '*.jpeg')
    img_paths=glob.glob(dirpath)
    f=open(os.path.join(datadir , labelpath))
    lines=f.readlines()
    ret_paths=[]
    ret_labels=[]
    for path in img_paths:
        img_name = os.path.split(path)[-1]
        img_name =os.path.splitext(img_name)[0]
        for line in lines:
            if line.split(',')[0] == img_name:
                ret_paths.append(path)
                ret_labels.append(line[1].replace('\n' , ''))
    ret_labels=map(int , ret_labels)
    tfrecord_path=os.path.join(datadir, 'fundus_{}.tfrecord'.format(i))
    print len(ret_paths)
    Dataprovider.Dataprovider.make_tfrecord_rawdata(tfrecord_path, img_sources=ret_paths, labels=ret_labels)









