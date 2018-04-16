#-*- coding:utf-8 -*-
import cnn
import tensorflow
import numpy as np
import cam
import aug
import tensorflow as tf
from DNN import DNN
from Recorder import Recorder

"""
def conv2d_with_bias(_input, out_feature, kernel_size, strides, padding):
    in_feature = int(_input.get_shape()[-1])
    kernel = weight_variable_msra([kernel_size, kernel_size, in_feature, out_feature], name='kernel')
    layer = tf.nn.conv2d(_input, kernel, strides, padding) + bias_variable(shape=[out_feature])
    layer = tf.nn.relu(layer)
    print layer
    return layer
def fc_with_bias(_input, out_features):
    in_fearues = int(_input.get_shape()[-1])
    kernel = weight_variable_xavier([in_fearues, out_features], name='kernel')
    layer = tf.matmul(_input, kernel) + bias_variable(shape=[out_features])
    print layer
    return layer

def avg_pool(_input, k):
    ksize = [1, k, k, 1]
    strides = [1, k, k, 1]
    padding = 'VALID'
    output = tf.nn.avg_pool(_input, ksize, strides, padding)
    return output

def fc_layer(_input, out_feature, act_func='relu', dropout='True'):
    assert len(_input.get_shape()) == 2, len(_input.get_shape())
    in_features = _input.get_shape()[-1]
    w = weight_variable_xavier([in_features, out_feature], name='W')
    b = bias_variable(shape=out_feature)
    layer = tf.matmul(_input, w) + b
    if act_func == 'relu':
        layer = tf.nn.relu(layer)
    return layer

def fc_layer_to_clssses(_input, n_classes):
    in_feature = int(_input.get_shape()[-1])
    W = weight_variable_xavier([in_feature, n_classes], name='W')
    bias = bias_variable([n_classes])
    logits = tf.matmul(_input, W) + bias
    return logits
"""

class VGG(DNN):
    def __init__(self , optimizer_name, use_bn, use_l2Loss , model , logit_type , datatype):
        DNN.initialize(optimizer_name , use_bn, use_l2Loss, model, logit_type, datatype)
        self.build_graph()
        DNN.algorithm(self.logits) # 이걸 self 로 바꾸면 안된다.
        DNN.sess_start()




    def build_graph(self):
        ##### define conv connected layer #######
        image_size = int(DNN.x_.get_shape()[-2])
        n_classes = int(DNN.y_.get_shape()[-1])
        if self.model == 'vgg_11':
            print 'Model : {}'.format('vgg 11')
            conv_out_features = [64, 128, 256, 256, 512, 512, 512, 512]
            conv_kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
            conv_strides = [1, 1, 1, 1, 1, 1, 1, 1]
            before_act_bn_mode = [False, False, False, False, False, False, False, False, False]
            after_act_bn_mode = [False, False, False, False, False, False, False, False, False]
            if self.use_BN == True:
                before_act_bn_mode = [True, True, True, True, True, True, True, True]
            allow_max_pool_indices = [0, 1, 2, 3, 5, 7]

        elif self.model == 'vgg_13':
            conv_out_features = [64, 64, 128, 128, 256, 256, 512, 512, 512, 512]
            conv_kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            conv_strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            before_act_bn_mode = [False, False, False, False, False, False, False, False, False, False]
            after_act_bn_mode = [False, False, False, False, False, False, False, False, False, False]
            if self.use_BN == True:
                before_act_bn_mode = [True, True, True, True, True, True, True, True, True, True]
            allow_max_pool_indices = [1, 3, 5, 7, 9]

        elif self.model == 'vgg_16':
            conv_out_features = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            conv_kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            conv_strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            before_act_bn_mode = [False, False, False, False, False, False, False, False, False, False, False, False, False]
            after_act_bn_mode = [False, False, False, False, False, False, False, False, False, False, False, False, False]
            if self.use_BN == True:
                before_act_bn_mode = [True, True, True, True, True, True, True, True, True, True, True, True, True]

            allow_max_pool_indices = [1, 3, 6, 9, 12]

        elif self.model == 'vgg_19':
            conv_out_features = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
            conv_kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            conv_strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            before_act_bn_mode = [False, False, False, False, False, False, False, False, False, False, False, False, False,
                                  False, False]
            after_act_bn_mode = [False, False, False, False, False, False, False, False, False, False, False, False, False,
                                 False, False]
            if self.use_BN == True:
                before_act_bn_mode = [True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                      True, True]
            allow_max_pool_indices = [1, 3, 7, 9, 11, 15]
        else:
            print '[vgg_11 , vgg_13 , vgg_16 , vgg_19]'
            raise AssertionError


        ###VGG Paper ###
        """
        VGG-11 64 max 128 max 256 256 max 512 512 max 512 512 max 4096 4096 1000  
        VGG-11 64 LRN max 128 max 256 256 max 512 512 max 512 512 max 4096 4096 1000
        VGG-13 64 64 LRN max 128 128 max 256 256 max 512 512 max 512 512 max 4096 4096 1000
        VGG-16 64 64 LRN max 128 128 max 256 256 256 max 512 512 512 max 512 512 512 max 4096 4096 1000
        VGG-16 64 64 LRN max 128 128 max 256 256 256 max 512 512 512 max 512 512 512 max 4096 4096 1000
        VGG-16 64 64 LRN max 128 128 max 256 256 256 256 max 512 512 512 512 max 512 512 512 512 max 4096 4096 1000
    
        """
        print '###############################################################'
        print '#                            {}'.format(self.model),'                          #'
        print '###############################################################'
        layer = DNN.x_
        for i in range(len(conv_out_features)):
            with tf.variable_scope('conv_{}'.format(str(i))) as scope:
                # Apply Batch Norm
                if before_act_bn_mode[i] == True:
                    layer = self.batch_norm_layer(layer, DNN.is_training , 'before_BN')
                # Apply Convolution
                layer = self.convolution2d(name=None, x=layer, out_ch=conv_out_features[i], k=conv_kernel_sizes[i],
                                           s=conv_strides[i], padding="SAME")
                # Apply Max Pooling Indices
                if i in allow_max_pool_indices:
                    print 'max pooling layer : {}'.format(i)
                    layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                    print layer
                layer = tf.nn.relu(layer)
                # Apply Batch Norm
                if after_act_bn_mode[i] == True:
                    layer = self.batch_norm_layer(layer, DNN.is_training , 'after_BN')
                # Apply Dropout layer=tf.nn.dropout(layer , keep_prob=conv_keep_prob)
                layer = tf.cond(DNN.is_training, lambda: tf.nn.dropout(layer, keep_prob=1.0), lambda: layer)

        self.top_conv = tf.identity(layer, name='top_conv')

        if self.logit_type =='gap':
            layer = self.gap(self.top_conv)
            self.logits = self.fc_layer_to_clssses(layer , self.n_classes)

        elif self.logit_type =='fc':
            fc_features=[4096 ,4096]
            before_act_bn_mode = [False, False]
            after_act_bn_mode = [False, False]
            self.top_conv = layer
            for i in range(len(fc_features)):
                with tf.variable_scope('fc_{}'.format(str(i))) as scope:
                    print i
                    if before_act_bn_mode[i]:
                        layer = self.batch_norm_layer(layer, self.is_training , 'bn')
                    layer = self.affine(name=None, x=layer, out_ch=fc_features[i], keep_prob=0.5 , is_training=self.is_training)
                    if after_act_bn_mode[i]:
                        layer = self.batch_norm_layer(layer, self.is_training, 'bn')
            self.logits=self.fc_layer_to_clssses(layer , self.n_classes)
        else:
            print '["fc", "gap"]'
            raise AssertionError
        return self.logits

def train_algorithm_momentum(logits, labels, learning_rate, use_nesterov, l2_loss):
    print 'Optimizer : Momentum'
    print 'Use Nesterov : ', use_nesterov
    print 'L2 Loss : ', l2_loss
    prediction = tf.nn.softmax(logits, name='softmax')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                                   name='cross_entropy')

    momentum = 0.9;
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum, use_nesterov=use_nesterov)
    if l2_loss:
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')
        weight_decay = 1e-4
        train_op = optimizer.minimize(cross_entropy + l2_loss * weight_decay, name='train_op')
    else:
        train_op = optimizer.minimize(cross_entropy, name='train_op')
    correct_prediction = tf.equal(
        tf.argmax(prediction, 1),
        tf.argmax(labels, 1), name='correct_prediction')

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name='accuracy')
    return train_op, accuracy, cross_entropy, prediction


def train_algorithm_adam(logits, labels, learning_rate, l2_loss):
    prediction = tf.nn.softmax(logits, name='softmax')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                                   name='cross_entropy')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    if l2_loss:
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')
        weight_decay = 1e-4
        train_op = optimizer.minimize(cross_entropy + l2_loss * weight_decay, name='train_op')
    else:
        train_op = optimizer.minimize(cross_entropy, name='train_op')
    correct_prediction = tf.equal(
        tf.argmax(prediction, 1),
        tf.argmax(labels, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name='accuracy')
    return train_op, accuracy, cross_entropy, prediction


def train_algorithm_grad(logits, labels, learning_rate, l2_loss):
    prediction = tf.nn.softmax(logits, name='softmax')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                                   name='cross_entropy')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    if l2_loss:
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')
        weight_decay = 1e-4
        train_op = optimizer.minimize(cross_entropy + l2_loss * weight_decay, name='train_op')
    else:
        train_op = optimizer.minimize(cross_entropy, name='train_op')
    correct_prediction = tf.equal(
        tf.argmax(prediction, 1),
        tf.argmax(labels, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name='accuracy')
    return train_op, accuracy, cross_entropy, prediction


if __name__ == '__main__':
    vgg=VGG('vgg_11', True , logit_type='fc' , n_classes=10)




