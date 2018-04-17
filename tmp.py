from Dataprovider import Input
import glob
import os
img_paths=glob.glob('/Users/seongjungkim/Downloads/train.zip/train/*.jpeg')
f= open('/Users/seongjungkim/Downloads/trainLabels.csv/trainLabels.csv')
lines = f.readlines()
names_labels={}
paths=[]
labels=[]
for line in lines:
    name , label = line.split(',')
    names_labels[name] = label

for img_path in img_paths:
    fname =os.path.split(img_path)[-1]
    name=os.path.splitext(fname)[0]
    paths.append(img_path)
    labels.append(names_labels[name].replace('\n' , ''))
labels=map(int , labels)
Input.make_tfrecord_rawdata('tmp.tfrecord' , paths , labels  )




Input.get_shuffled_batch('tmp.tfrecord' , batch_size=60 , resize=(224,224))