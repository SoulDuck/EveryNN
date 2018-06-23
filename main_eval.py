import tensorflow as f
import Tester
from Dataprovider import Dataprovider
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import cls2onehot



restore_model = '/home/mediwhale-5/PythonProjects/everyNN/models/resnet_18/4/model-26730'
dirpath = '/home/mediwhale-5/PythonProjects/everyNN/my_data/project10'

restore_model = './models/best_models/exam_id_000000065/model-37719'
dirpath = './my_data/0100-0000003-019'


test_data = os.path.join(dirpath ,'test_0_10_11_inf.tfrecord')
val_data = os.path.join(dirpath ,'val_0_10_11_inf.tfrecord')

tester=Tester.Tester(None)
tester._reconstruct_model(restore_model)

#tester.validate( save_model = None )
test_imgs  , test_labs , test_fs = Dataprovider.reconstruct_tfrecord_rawdata(test_data , None )
val_imgs ,val_labs , val_fs = Dataprovider.reconstruct_tfrecord_rawdata(val_data , None)

val_labs=cls2onehot(val_labs ,2)
test_labs=cls2onehot(test_labs ,2)
"""
for i,img in enumerate(test_imgs):
    plt.imshow(test_imgs[0])
    plt.show()
    plt.imsave('tmp/{}.jpg'.format(i) , img)
"""
tester.n_classes =2
val_imgs=val_imgs/255.
test_imgs= test_imgs/255.
print len(val_imgs)
print len(test_imgs)
plt.imsave('delete.png',test_imgs[0])
tester.validate(test_imgs , test_labs , 60 ,0 ,False)
print ''
print tester.acc
print tester.loss
print tester.acc_by_labels
for pred in tester.pred_all:
    print pred

"""
for pred in tester.pred_all:
    print pred
for lab in val_labs:
    print lab
"""






