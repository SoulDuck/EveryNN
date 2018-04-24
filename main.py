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
vgg = VGG('sgd' , False , True,   model_name, 'gap'  , 'cifar10' , 60 , resize=(32,32) , num_epoch=100)
# 350x350 이미지를 random crop 해 300x300 으로 학습 합니다.
recorder = Recorder(folder_name=model_name)
trainer = Trainer(recorder)
tester=Tester(recorder)


global_step=0
for i in range(1000):
    global_step = trainer.training(1000,global_step)
    mean_acc, mean_loss, pred_all=tester.validate_tfrecord(cifar.test_tfrecords[0] , None , None )
    print '\t Acc :{}  Loss :{} '.format(mean_acc , mean_loss)



