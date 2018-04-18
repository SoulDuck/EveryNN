# bring kaggle fundus
import Dataprovider
import os , glob
import sys
import argparse
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
        print path
        img_name = os.path.split(path)[-1]
        img_name =os.path.splitext(img_name)[0]
        for line in lines:

            if img_name in line.split(',')[0]:
                ret_paths.append(path)
                ret_labels.append(line[1].replace('\n' , ''))
        tfrecord_path=os.path.join(datadir, 'fundus_{}.tfrecord'.format(i))
        Dataprovider.Dataprovider.make_tfrecord_rawdata(tfrecord_path, img_sources=ret_paths, labels=ret_paths)









