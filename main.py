#-*- coding:utf-8 -*-
from VGG import VGG
from INCEPTION_V4 import INCEPTION_V4
from RESNET_V1 import RESNET_V1
from Recorder import Recorder
from Trainer import Trainer
from Tester import Tester
import numpy as np
import tensorflow as tf
import cifar
import my_data
import utils

#test_imgs , test_labs =my_data.get_test_imgs_labs((350,350))

#print test_labs

"""
model_name = 'vgg_11'
vgg = VGG('sgd' , False , True,   model_name, 'gap'  , 'cifar10' , 60 , resize=(350,350) , num_epoch=100)
"""

model_name = 'resnet_18'
batch_size = 45
resnet_v1=RESNET_V1('adam' , True , True , logit_type='gap' , datatype= 'my_data' ,batch_size=45 , resize=(350,350), num_epoch=10\
          ,init_lr = 7e-5 , lr_decay_step= 100 ,model = model_name)
recorder = Recorder(folder_name=model_name)
trainer = Trainer(recorder ,train_iter= 100)
tester=Tester(recorder)
test_imgs, test_labs ,fnames =resnet_v1.dataprovider.reconstruct_tfrecord_rawdata(resnet_v1.dataprovider.test_tfrecord_path , None)
test_labs=utils.cls2onehot(test_labs, 2)

print np.shape(test_imgs)
print np.shape(test_labs)
print np.shape(fnames)

for i in range(1):
    #val_acc, val_loss, val_preds = tester.validate_tfrecords(my_data.test_tfrecord_path, None, None)
    tester.validate(test_imgs[:2] ,test_labs[:2] ,batch_size , trainer.train_step)
    recorder.write_acc_loss('Validation test', tester.loss, tester.acc, trainer.train_step)
    tester.show_acc_loss(trainer.train_step)
    #abnormal_acc, abnormal_loss, abnormal_preds= tester.validate_tfrecord(my_data.test_abnormal_tfrecord, None, None ,global_step )
    #print 'Acc : {} Loss : {}'.format((normal_acc+abnormal_acc)/2. , (normal_loss+abnormal_loss)/2.)
    global_step = trainer.training()
resnet_v1.sess_stop()