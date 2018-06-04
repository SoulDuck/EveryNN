# -*- coding:utf-8 -*-
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
# test_imgs , test_labs =my_data.get_test_imgs_labs((350,350))
# print test_labs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--optimizer_name', type=str, choices=['sgd', 'momentum', 'adam'])
parser.add_argument('--batch_size', type=int)
parser.add_argument('--l2_weight_decay', type=int)

parser.add_argument('--BN', dest='use_BN', action='store_true')
parser.add_argument('--no_BN', dest='use_BN', action='store_false')

parser.add_argument('--logit_type')

parser.add_argument('--datatype')

parser.add_argument('--cropped_size', type=int)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--init_lr', type=float)
parser.add_argument('--lr_decay_step', type=int)

parser.add_argument('--model', type=str)

parser.add_argument('--aug_list', nargs='+', type=str, default=['aug_lv0', 'aug_lv1', 'aug_rotate', 'aug_clahe'])

args = parser.parse_args()

"""
model_name = 'vgg_11'
vgg = VGG('sgd' , False , True,   model_name, 'gap'  , 'cifar10' , 60 , resize=(350,350) , num_epoch=100)
"""

model_name = 'resnet_18'
batch_size = 30
resnet_v1 = RESNET_V1(args.optimizer_name, args.use_bn, args.l2_weight_decay, args.logit_type, args.datatype,
                      args.batch_size, args.cropped_size, args.num_epoch,
                      args.init_lr, args.lr_decay_step, args.model, args.aug_list))


recorder = Recorder(folder_name=model_name)
trainer = Trainer(recorder, train_iter=100)
tester = Tester(recorder)

test_imgs, test_labs, fnames = resnet_v1.dataprovider.reconstruct_tfrecord_rawdata(
    resnet_v1.dataprovider.test_tfrecord_path, None)
test_labs = utils.cls2onehot(test_labs, 2)
test_imgs = test_imgs / 255.

val_imgs, val_labs, fnames = resnet_v1.dataprovider.reconstruct_tfrecord_rawdata(
    resnet_v1.dataprovider.val_tfrecord_path, None)
val_labs = utils.cls2onehot(val_labs, 2)
val_imgs = val_imgs / 255.

print np.shape(test_imgs)
print np.shape(test_labs)
print np.shape(fnames)

for i in range(10):
# val_acc, val_loss, val_preds = tester.validate_tfrecords(my_data.test_tfrecord_path, None, None)
    tester.validate(val_imgs[:], val_labs[:], batch_size, trainer.train_step)
recorder.write_acc_loss('Validation test', tester.loss, tester.acc, trainer.train_step)
tester.show_acc_loss(trainer.train_step)
global_step = trainer.training()
resnet_v1.sess_stop()
