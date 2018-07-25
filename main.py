#-*- coding:utf-8 -*-
from VGG import VGG
from INCEPTION_V4 import INCEPTION_V4
from Densenet import Densenet
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
import aug

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
if 'resnet' in args.model_name:
    cnn_model=RESNET_V1(args.opt , args.use_bn , args.l2_weight_decay, args.logit_type , args.datatype ,args.batch_size, args.cropped_size,\
                    args.num_epoch ,args.init_lr, args.lr_decay_step, args.model_name ,args.aug_list)
elif 'vgg' in args.model_name:
    cnn_model=VGG(args.opt , args.use_bn , args.l2_weight_decay, args.logit_type , args.datatype ,args.batch_size, args.cropped_size,\
                    args.num_epoch ,args.init_lr, args.lr_decay_step, args.model_name ,args.aug_list)
elif 'inception' in args.model_name:
    pass;
elif 'densenet' in args.model_name:
    cnn_model=Densenet(args.opt , args.use_bn , args.l2_weight_decay, args.logit_type , args.datatype ,args.batch_size, args.cropped_size,\
                    args.num_epoch ,args.init_lr, args.lr_decay_step, args.model_name ,args.aug_list)
else:
    pass;

recorder = Recorder(folder_name=args.model_name)
trainer = Trainer(recorder ,train_iter = 100 )
tester=Tester(recorder)

# Reconstruct Test , Validation Data
test_imgs, test_labs ,fnames =cnn_model.dataprovider.reconstruct_tfrecord_rawdata(cnn_model.dataprovider.test_tfrecord_path , None)
test_labs=utils.cls2onehot(test_labs, cnn_model.n_classes)

val_imgs, val_labs ,fnames =cnn_model.dataprovider.reconstruct_tfrecord_rawdata(cnn_model.dataprovider.val_tfrecord_path , None)
val_labs=utils.cls2onehot(val_labs, cnn_model.n_classes)


if 'aug_clahe' in args.aug_list:
    print "Clahe is applied , Validation images , Test Images "
    test_imgs=aug.apply_clahe(test_imgs)
    val_imgs = aug.apply_clahe(val_imgs)

if 'aug_projection' in args.aug_list:
    print "Projection is applied , Validation images , Test Images "
    test_imgs=aug.fundus_projection(test_imgs)
    val_imgs = aug.fundus_projection(val_imgs)

if np.max(test_imgs) > 1 :
    test_imgs = test_imgs / 255.
if np.max(val_imgs) > 1:
    val_imgs = val_imgs / 255.

max_step=int(cnn_model.max_iter)
print 'Start Training , Max step : {}'.format(max_step)
for i in range(max_step):
    #val_acc, val_loss, val_preds = tester.validate_tfrecords(my_data.test_tfrecord_path, None, None)
    tester.validate(test_imgs[:] ,test_labs[:] ,args.batch_size , trainer.train_step)
    tester.show_acc_loss(trainer.train_step)
    tester.show_acc_by_label()
    global_step = trainer.training(args.aug_list)
cnn_model.sess_stop()


