#-*- coding:utf-8 -*-
import numpy  as np
import Dataprovier
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

class DNN(object):

    #define class variable
    x_=None
    y_=None
    cam_ind = None
    learning_rate = None
    is_training = None

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())


    def weight_variable_xavier(self,shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def convolution2d(self,name,x,out_ch,k=3 , s=2 , padding='SAME'):
        def _fn():
            in_ch = x.get_shape()[-1]
            filter = tf.get_variable("w", [k, k, in_ch, out_ch],
                                     initializer=tf.contrib.layers.variance_scaling_initializer())
            bias = tf.Variable(tf.constant(0.1), out_ch, name='b')
            layer = tf.nn.conv2d(x, filter, [1, s, s, 1], padding) + bias
            layer = tf.nn.relu(layer, name='relu')
            if __debug__ == True:
                print 'layer name : ', name
                print 'layer shape : ', layer.get_shape()

            return layer

        if name is not None:
            with tf.variable_scope(name) as scope:
                layer = _fn()
        else:
            layer = _fn()
        return layer

    def max_pool(self,name, x, k=3, s=2, padding='SAME'):
        with tf.variable_scope(name) as scope:
            if __debug__ == True:
                layer = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=padding)
                print 'layer name :', name
                print 'layer shape :', layer.get_shape()
        return layer
    def avg_pool(self,name, x, k=3, s=2, padding='SAME'):
        with tf.variable_scope(name) as scope:
            if __debug__ == True:
                layer = tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=padding)
                print 'layer name :', name
                print 'layer shape :', layer.get_shape()
        return layer

    def batch_norm_layer(self,x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
                              updates_collections=None,
                              is_training=True,
                              reuse=None,  # is this right?
                              trainable=True,
                              scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
                                  updates_collections=None,
                                  is_training=False,
                                  reuse=True,  # is this right?
                                  trainable=True,
                                  scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


    """
    다른 버전의 batch norm
    @classmethod
    def batch_norm(_input, is_training):
        output = tf.contrib.layers.batch_norm(_input, scale=True, \
                                              is_training=is_training, updates_collections=None)
        return output
    """
    def dropout(self, _input, is_training, keep_prob=0.5):
        if keep_prob < 1:
            output = tf.cond(is_training, lambda: tf.nn.dropout(_input, keep_prob), lambda: _input)
        else:
            output = _input
        return output

    def affine(self,name, x, out_ch, keep_prob , is_training):
        def _fn(x):
            if len(x.get_shape()) == 4:
                batch, height, width, in_ch = x.get_shape().as_list()
                w_fc = tf.get_variable('w', [height * width * in_ch, out_ch],
                                       initializer=tf.contrib.layers.xavier_initializer())
                x = tf.reshape(x, (-1, height * width * in_ch))
            elif len(x.get_shape()) == 2:
                batch, in_ch = x.get_shape().as_list()
                w_fc = tf.get_variable('w', [in_ch, out_ch], initializer=tf.contrib.layers.xavier_initializer())
            else:
                print 'x n dimension must be 2 or 4 , now : {}'.format(len(x.get_shape()))
                raise AssertionError
            print x
            b_fc = tf.Variable(tf.constant(0.1), out_ch)
            layer = tf.matmul(x, w_fc) + b_fc

            layer = tf.nn.relu(layer)
            layer =self.dropout(layer ,is_training , keep_prob)
            print 'layer name :'
            print 'layer shape :', layer.get_shape()
            print 'layer dropout rate :', keep_prob
            return layer

        if name is not None:
            with tf.variable_scope(name) as scope:
                layer = _fn(x)
        else:
            layer = _fn(x)
        return layer

    def gap(self, x):
        gap = tf.reduce_mean(x, (1, 2))
        return gap
    def fc_layer_to_clssses(self, layer , n_classes):
        #layer should be flatten
        assert len(layer.get_shape()) ==2
        in_ch=int(layer.get_shape()[-1])
        with tf.variable_scope('logits') as scope:
            w = tf.get_variable('w', shape=[in_ch, n_classes], initializer=tf.random_normal_initializer(0, 0.01),
                                    trainable=True)
            b = tf.Variable(tf.constant(0.1), n_classes , name='b')
        logits = tf.matmul(layer, w, name='logits') +b
        return logits

    """
    다른 버전의 batch norm
    @classmethod
    def batch_norm(_input, is_training):
        output = tf.contrib.layers.batch_norm(_input, scale=True, \
                                              is_training=is_training, updates_collections=None)
        return output
    """
    @classmethod
    def algorithm(cls , optimizer='GradientDescentOptimizer'):
        """
        :param y_conv: logits
        :param y_: labels
        :param learning_rate: learning rate
        :return:  pred,pred_cls , cost , correct_pred ,accuracy
        """
        if __debug__ == True:
            print 'debug start : cnn.py | algorithm'
            print 'optimizer option : GradientDescentOptimizer(default) | AdamOptimizer | moment | '
            print 'selected optimizer : ', optimizer
            print cls.logits.get_shape()
            print cls.y_.get_shape()
        optimizer_dic = {'GradientDescentOptimizer': tf.train.GradientDescentOptimizer,
                         'AdamOptimizer': tf.train.AdamOptimizer}

        cls.pred = tf.nn.softmax(cls.logits, name='softmax')
        cls.pred_cls = tf.argmax(cls.pred, axis=1, name='pred_cls')
        cls.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cls.logits, labels=cls.y_), name='cost')
        cls.train_op = optimizer_dic[optimizer](cls.learning_rate).minimize(cls.cost)
        cls.correct_pred = tf.equal(tf.argmax(cls.logits, 1), tf.argmax(cls.y_, 1), name='correct_pred')
        cls.accuracy = tf.reduce_mean(tf.cast(cls.correct_pred, dtype=tf.float32), name='accuracy')


    @classmethod
    def _define_input(cls, shape):
        cls.x_= tf.placeholder(tf.float32, shape=shape, name='x_')
        cls.y_ = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='y_')
        cls.cam_ind = tf.placeholder(tf.int32, shape=[], name='cam_ind')
        cls.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        cls.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

    @classmethod
    def _sess_start(self):
        sess = tf.Session()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        return sess

    @classmethod
    def initialize(cls, optimizer_name, use_BN, use_l2_loss, model, logit_type, datatype):
        cls.optimizer_name = optimizer_name
        cls.use_BN = use_BN
        cls.use_l2_loss = use_l2_loss
        cls.vgg_model = model
        cls.logit_type = logit_type

        ## input pipeline
        cls.pipeline = Dataprovier.Input(datatype)
        batch_xs, batch_ys, _ = cls.pipeline.next_batch(10)
        _, h, w, ch = np.shape(batch_xs)
        _, n_classes = np.shape(batch_ys)
        cls._define_input(shape=[None, h, w, ch]) #
    @classmethod
    def build_graph(cls):
        raise NotImplementedError
    """
    @classmethod
    def build_model(cls , model):
        ## build VGG or Densenet or Resnet
        if model == 'vgg_11' or model == 'vgg_13' or model == 'vgg_16' or model == 'vgg_19':
            print 'model type : {}'.format(model)
            print cls.x_
            cls.top_conv=VGG(model , bn=True)
    """


if __name__ == '__main__':
    dnn=DNN()
    dnn.initialize('sgd',True , True , 'vgg_11' , logit_type='fc' , datatype='cifar10')
    #cls._algorithm(cls.optimizer_name)
