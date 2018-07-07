import tensorflow as f
import Tester
from Dataprovider import Dataprovider
import numpy as np
import matplotlib.pyplot as plt
import random
from cam import inspect_cam
import os , glob
from utils import cls2onehot
from PIL import Image


restore_model = '/home/mediwhale-5/PythonProjects/everyNN/models/resnet_18/4/model-26730'
dirpath = '/home/mediwhale-5/PythonProjects/everyNN/my_data/project10'

restore_model = './models/best_models/exam_id_000000065/model-37719'
dirpath = './my_data/0100-0000003-019'

restore_model  = '/home/mediwhale-5/PythonProjects/everyNN/models/resnet_18/10/model-26631'
dirpath = '/home/mediwhale-5/PythonProjects/cac_regressor/0100-0000003-023/0100-0000003-022'

restore_model  = '/home/mediwhale/PycharmProjects/everyNN/models/resnet_34/4/model-29700'
#dirpath = '/home/mediwhale-5/PythonProjects/cac_regressor/0100-0000003-023/0100-0000003-022'
"""
test_data = os.path.join(dirpath ,'test_0_10_11_inf.tfrecord')
val_data = os.path.join(dirpath ,'val_0_10_11_inf.tfrecord')

#tester.validate( save_model = None )
test_imgs  , test_labs , test_fs = Dataprovider.reconstruct_tfrecord_rawdata(test_data , None )
val_imgs ,val_labs , val_fs = Dataprovider.reconstruct_tfrecord_rawdata(val_data , None)
print test_labs
print val_labs
val_labs=cls2onehot(val_labs ,2)
test_labs=cls2onehot(test_labs ,2)



for i,img in enumerate(val_imgs):
    plt.imshow(test_imgs[0])
    plt.show()
    plt.imsave('tmp/{}.jpg'.format(i) , img)

exit()
"""


paths_0=glob.glob('./images/0100-0000003-019_label_0/*.png')
paths_0  = paths_0 #+ paths_0[-50:]
paths_0=sorted(paths_0)
paths_1=glob.glob('./images/0100-0000003-019_label_1/*.png')
paths_1  = paths_1 #+ paths_1[-50:]
paths_1=sorted(paths_1)
imgs= []
# Label
n_label_0=len(paths_0)
n_label_1=len(paths_1)
print (n_label_1)
print (n_label_0)
test_labs=np.zeros([n_label_0 + n_label_1 , 2])
test_labs[:n_label_0 , 0] = 1
test_labs[n_label_0 : ,1 ] = 1
test_cls = np.argmax(test_labs,axis=1)

print len(paths_0)
print len(paths_1)

# Image
for i,path in enumerate(paths_0 + paths_1):
    print i,os.path.split(path)[-1]
    img=np.asarray(Image.open(path).convert('RGB'))
    imgs.append(img)
test_imgs=np.asarray(imgs)

assert len(test_labs) == len(test_imgs)
print np.shape(test_imgs)





tester=Tester.Tester(None)
tester._reconstruct_model(restore_model)
tester.n_classes =2
test_imgs = test_imgs/255.
tester.validate(test_imgs , test_labs, 60 ,0 ,False)

for i,path in enumerate(paths_0 + paths_1):
    print os.path.split(path)[-1].split('_')[0] , tester.pred_all[i] ,test_cls[i]

inspect_cam(tester.sess , tester.classmap_op  , tester.top_conv ,test_imgs , test_labs , 0 , tester.x_ , tester.y_ , tester.is_training , tester.logits_ )
actmap=tester.sess.run(tester.classmap_op , {tester.x_: test_imgs[0:1] , tester.is_training :False})
actmap=np.squeeze(actmap)
print np.shape(actmap)
plt.imsave('actmap_0.png' , actmap)

exit()

print ''
print tester.acc
print tester.loss
print tester.acc_by_labels

for pred in tester.pred_all:
    print pred


test_cls=np.argmax(test_labs , axis=1)
predStrength=np.asarray(tester.pred_all)[:,1]




"""
indices = np.where([predStrength > 0.5])[1]
rev_indices = np.where([predStrength < 0.5])[1]

predStrength_ = list(predStrength[indices[:160]]) + list(predStrength[rev_indices[:40]])
test_cls_=list(test_cls[indices[:160]]) + list(test_cls[rev_indices[:40]])
"""

tester.plotROC(predStrength=predStrength, labels= test_cls ,  prefix='CAC fundus classifier' , savepath='tmp.png')
"""

for pred in tester.pred_all:
    print pred
for lab in val_labs:
    print lab
tester.ensemble(test_imgs, test_labs, 60, './models/best_models/0_from_5555/model-37719', './models/best_models/0_from_5566/model-14751',
                './models/best_models/0_from_5571/model-45144', './models/best_models/1_from_5555/model-25839',
                './models/best_models/1_from_5571/model-64548')


"""


pred_cls =np.argmax(tester.pred_all , axis= 1)

for i in range(len(pred_cls)):
    if pred_cls[i] == test_cls[i]:
        if pred_cls[i]==0:
            plt.imsave('./images/TT/{}.png'.format(i),test_imgs[i])
        elif pred_cls[i]==1:
            plt.imsave('./images/FF/{}.png'.format(i), test_imgs[i])
    else:
        if pred_cls[i]==0:
            plt.imsave('./images/TF/{}.png'.format(i), test_imgs[i])
        elif pred_cls[i] ==1:
            plt.imsave('./images/FT/{}.png'.format(i), test_imgs[i])


