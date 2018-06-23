import tensorflow as f
import Tester
from Dataprovider import Dataprovider
import numpy as np
import matplotlib.pyplot as plt
import os , glob
from utils import cls2onehot
from PIL import Image


restore_model = '/home/mediwhale-5/PythonProjects/everyNN/models/resnet_18/4/model-26730'
dirpath = '/home/mediwhale-5/PythonProjects/everyNN/my_data/project10'

restore_model = './models/best_models/exam_id_000000065/model-37719'
dirpath = './my_data/0100-0000003-019'

"""
test_data = os.path.join(dirpath ,'test_0_10_11_inf.tfrecord')
val_data = os.path.join(dirpath ,'val_0_10_11_inf.tfrecord')

#tester.validate( save_model = None )
test_imgs  , test_labs , test_fs = Dataprovider.reconstruct_tfrecord_rawdata(test_data , None )
val_imgs ,val_labs , val_fs = Dataprovider.reconstruct_tfrecord_rawdata(val_data , None)



for i,img in enumerate(test_imgs):
    plt.imshow(test_imgs[0])
    plt.show()
    plt.imsave('tmp/{}.jpg'.format(i) , img)

exit()
"""

paths_0=glob.glob('./images/0100-0000003-019_label_0/*.png')
paths_0=sorted(paths_0)
paths_1=glob.glob('./images/0100-0000003-019_label_1/*.png')
paths_1=sorted(paths_1)
imgs= []
# Label
n_label_0=len(paths_0)
n_label_1=len(paths_1)
print (n_label_1)
print (n_label_0)
test_labs=np.zeros([n_label_0 + n_label_1 , 2])
test_labs[:n_label_0 , 0]=1
test_labs[n_label_0 : ,1 ]=1

print test_labs

# Image
for path in paths_0 + paths_1:
    print path
    img=np.asarray(Image.open(path).convert('RGB'))
    imgs.append(img)
test_imgs=np.asarray(imgs)

assert len(test_labs) == len(test_imgs)
print np.shape(test_imgs)



tester=Tester.Tester(None)
tester._reconstruct_model(restore_model)
tester.n_classes =2
test_imgs= test_imgs/255.
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

tester.ensemble(test_imgs, test_labs, 60, './models/best_models/0_from_5555', './models/best_models/0_from_5566',
                './models/best_models/0_from_5571', './models/best_models/1_from_5555',
                './models/best_models/1_from_5571')



