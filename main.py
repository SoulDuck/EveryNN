#-*- coding:utf-8 -*-
from VGG import VGG
from INCEPTION_V4 import INCEPTION_V4
from RESNET_V1 import RESNET_V1
from Recorder import Recorder
from Trainer import Trainer
from Tester import Tester
import tensorflow as tf
import cifar
import my_data
import utils

#test_imgs , test_labs =my_data.get_test_imgs_labs((350,350))

#print test_labs

model_name = 'vgg_11'
vgg = VGG('sgd' , False , True,   model_name, 'gap'  , 'mydata' , 60 , resize=(350,350) , num_epoch=100)
# 350x350 이미지를 random crop 해 300x300 으로 학습 합니다.
recorder = Recorder(folder_name=model_name)
trainer = Trainer(recorder)
tester=Tester(recorder)


global_step=0
for i in range(10000):
    val_acc, val_loss, val_preds=tester.validate_tfrecord(my_data.test_tfrecords , None , None , global_step )
    #abnormal_acc, abnormal_loss, abnormal_preds= tester.validate_tfrecord(my_data.test_abnormal_tfrecord, None, None ,global_step )
    print ''
    print 'Validation Acc : {} | Loss : {}'.format(val_acc, val_loss)
    #print 'abnormal Acc : {} | abnormal Loss : {}'.format(abnormal_acc, abnormal_loss)
    #print 'Acc : {} Loss : {}'.format((normal_acc+abnormal_acc)/2. , (normal_loss+abnormal_loss)/2.)

    global_step = trainer.training(100,global_step)




