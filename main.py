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
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--datatype', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--use_bn', )
parser.add_argument('--l2_weight_decay', type=float)
parser.add_argument('--logit_type', type=str)
parser.add_argument('--num_epoch' , type=int)
parser.add_argument('--resize' , type=int)
parser.add_argument('--opt' , type=str)
parser.add_argument('--init_lr' , type=float)
parser.add_argument('--lr_decay_step' ,type=int)
parser.add_argument('')

args=parser.parse_args()
print args.batch_size
print args.datatype
print args.model_name
print args.use_bn
print args.l2_weight_decay
print args.logit_type
print args.data
print args.num_epoch
print args.resize
print args.opt



#test_imgs , test_labs =my_data.get_test_imgs_labs((350,350))

#print test_labs

"""
model_name = 'vgg_11'
vgg = VGG('sgd' , False , True,   model_name, 'gap'  , 'cifar10' , 60 , resize=(350,350) , num_epoch=100)
"""
resize=(args.resize ,args.resize)
model_name = 'resnet_18'
batch_size = args.batch_size
resnet_v1=RESNET_V1(args.opt , args.use_bn , args.l2_weight_decay, args.logit_type , args.data ,args.batch_size, resize,\
                    args.num_epoch ,args.init_lr, args.lr_decay_step, args.model )
recorder = Recorder(folder_name=model_name)
trainer = Trainer(recorder ,train_iter= 100)
tester=Tester(recorder)

test_imgs, test_labs ,fnames =resnet_v1.dataprovider.reconstruct_tfrecord_rawdata(resnet_v1.dataprovider.test_tfrecord_path , None)
test_labs=utils.cls2onehot(test_labs, 2)
test_imgs=test_imgs/255.

print np.shape(test_imgs)
print np.shape(test_labs)
print np.shape(fnames)

for i in range(10):
    #val_acc, val_loss, val_preds = tester.validate_tfrecords(my_data.test_tfrecord_path, None, None)
    tester.validate(test_imgs[:] ,test_labs[:] ,batch_size , trainer.train_step)
    recorder.write_acc_loss('Validation test', tester.loss, tester.acc, trainer.train_step)
    tester.show_acc_loss(trainer.train_step)
    global_step = trainer.training()
resnet_v1.sess_stop()