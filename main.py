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
parser.add_argument('--BN',dest='use_bn', action='store_true')
parser.add_argument('--no_BN',dest='use_bn', action='store_false')
parser.add_argument('--l2_weight_decay', type=float)
parser.add_argument('--logit_type', type=str)
parser.add_argument('--num_epoch' , type=int)
parser.add_argument('--cropped_size' , type=int)
parser.add_argument('--opt' , type=str)
parser.add_argument('--init_lr' , type=float)
parser.add_argument('--lr_decay_step' ,type=int)
parser.add_argument('--aug_list' ,nargs='+', type=str, default=['aug_lv0' ])# option 'aug_lv1' , 'aug_rotate' , 'aug_clahe'

args=parser.parse_args()
print 'batch_size : ' , args.batch_size
print 'datatype : ' , args.datatype
print 'model_name : ,',args.model_name
print 'use bn : , ',args.use_bn
print 'L2 weight Decay : ' , args.l2_weight_decay
print 'Logit Type : ' , args.logit_type
print 'Num Epoch  : ', args.num_epoch
print 'cropped_size:' , args.cropped_size
print 'Optimzer : ',args.opt
print 'Inital Learning Rate : ',args.init_lr
print 'Learning Rage Decay Step : ' , args.lr_decay_step
print 'Augmentation list : ', args.aug_list

#test_imgs , test_labs =my_data.get_test_imgs_labs((350,350))
#print test_labs

# Cifar VGG 11
"""
model_name = 'vgg_11'
vgg = VGG(args.opt , args.use_bn , args.l2_weight_decay, args.logit_type , args.datatype ,args.batch_size, args.cropped_size,\
                    args.num_epoch ,args.init_lr, args.lr_decay_step, args.model_name ,args.aug_list)
test_imgs, test_labs ,fnames =vgg.dataprovider.reconstruct_tfrecord_rawdata(vgg.dataprovider.test_tfrecord_path , None)
test_labs=utils.cls2onehot(test_labs , vgg.n_classes)
recorder = Recorder(folder_name=args.model_name)
trainer = Trainer(recorder ,train_iter = 100 )
tester=Tester(recorder)

if np.max(test_imgs) > 1 :
    test_imgs = test_imgs / 255.


max_step=int(vgg.max_iter*args.num_epoch/args.batch_size)
print 'Start Training , Max step : {}'.format(max_step)
for i in range(max_step):
    #val_acc, val_loss, val_preds = tester.validate_tfrecords(my_data.test_tfrecord_path, None, None)
    tester.validate(test_imgs[:10] ,test_labs[:10] ,args.batch_size , trainer.train_step)
    tester.validate_top_k(tester.pred_all , test_labs[:10] , 3,show_flag=True  )

    tester.show_acc_loss(trainer.train_step)
    tester.show_acc_by_label()
    global_step = trainer.training(args.aug_list)

vgg.sess_stop()
"""

# RESNET

resnet_v1=RESNET_V1(args.opt , args.use_bn , args.l2_weight_decay, args.logit_type , args.datatype ,args.batch_size, args.cropped_size,\
                    args.num_epoch ,args.init_lr, args.lr_decay_step, args.model_name ,args.aug_list)

recorder = Recorder(folder_name=args.model_name)
trainer = Trainer(recorder ,train_iter = 100 )
tester=Tester(recorder)

test_imgs, test_labs ,fnames =resnet_v1.dataprovider.reconstruct_tfrecord_rawdata(resnet_v1.dataprovider.test_tfrecord_path , None)
test_labs=utils.cls2onehot(test_labs, resnet_v1.n_classes)
val_imgs, val_labs ,fnames =resnet_v1.dataprovider.reconstruct_tfrecord_rawdata(resnet_v1.dataprovider.val_tfrecord_path , None)
val_labs=utils.cls2onehot(val_labs, resnet_v1.n_classes)

test_imgs_10_99 = np.load('./my_data/project10/cac_10_99_test.npy')
test_labs_10_99 = np.zeros([len(test_imgs_10_99 ) ,  2])
test_labs_10_99[: , 1]=1
val_imgs_10_99 = np.load('./my_data/project10/cac_10_99_val.npy')

if np.max(test_imgs) > 1 :
    test_imgs = test_imgs / 255.
if np.max(val_imgs) > 1:
    val_imgs = val_imgs / 255.

if np.max(test_imgs_10_99) > 1 :
    test_imgs_10_99 = test_imgs_10_99 / 255.
if np.max(val_imgs_10_99) > 1:
    val_imgs_10_99 = val_imgs_10_99 / 255.


max_step=int(resnet_v1.max_iter*args.num_epoch/args.batch_size)
print 'Start Training , Max step : {}'.format(max_step)
for i in range(max_step):
    #val_acc, val_loss, val_preds = tester.validate_tfrecords(my_data.test_tfrecord_path, None, None)

    tester.validate(test_imgs[:] ,test_labs[:] ,args.batch_size , trainer.train_step)
    tester.show_acc_loss(trainer.train_step)
    tester.show_acc_by_label()
    tester.validate(test_imgs_10_99[:], test_labs_10_99[:], args.batch_size, trainer.train_step , False ,)
    tester.show_acc_loss(trainer.train_step)
    tester.show_acc_by_label()
    global_step = trainer.training(args.aug_list)
resnet_v1.sess_stop()
