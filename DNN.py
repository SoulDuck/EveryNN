#-*- coding:utf-8 -*-
import numpy  as np
from Dataprovider import Dataprovider
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

class DNN(object):

    #define class variable
    #share variable
    x_=None
    y_=None
    cam_ind = None
    lr_ = None
    is_training = None
    pred=None
    pred_cls=None
    cost=None
    train_op=None
    correct_pred=None
    accuracy=None
    global_step=None
    n_classes=None
    optimizer_name = None
    init_lr=None
    lr_decay_step=None
    l2_weight_decay=None
    num_epoch=None
    max_iter = None


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

    def convolution2d_manual(self,name, x, out_ch, k_h, k_w, s=2, padding='SAME'):
        def _fn():
            in_ch = x.get_shape()[-1]
            filter = tf.get_variable("w", [k_h, k_w, in_ch, out_ch], initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.Variable(tf.constant(0.1), out_ch)
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

    def batch_norm_layer(self , x, phase_train, scope_bn):
        with tf.variable_scope(scope_bn):
            n_out = int(x.get_shape()[-1])
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)
            if len(x.get_shape()) == 4:  # for convolution Batch Normalization
                print 'BN for Convolution was applied'
                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            if len(x.get_shape()) == 2:  # for Fully Convolution Batch Normalization:
                print 'BN for FC was applied'
                batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

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
        gap = tf.reduce_mean(x, (1, 2) , name='gap')
        return gap
    def fc_layer_to_clssses(self, layer , n_classes):
        #layer should be flatten
        assert len(layer.get_shape()) ==2
        in_ch=int(layer.get_shape()[-1])
        with tf.variable_scope('final') as scope:
            w = tf.get_variable('w', shape=[in_ch, n_classes], initializer=tf.random_normal_initializer(0, 0.01),
                                    trainable=True)
            b = tf.Variable(tf.constant(0.1), n_classes , name='b')
            logits = tf.matmul(layer, w, name='matmul') +b
        logits=tf.identity(logits , name='logits')
        return logits

    def get_class_map(self,name, x, cam_ind, im_width , w=None):
        out_ch = int(x.get_shape()[-1])
        conv_resize = tf.image.resize_bilinear(x, [im_width, im_width])
        if w is None:
            with tf.variable_scope(name, reuse=True) as scope:
                label_w = tf.gather(tf.transpose(tf.get_variable('w')), cam_ind)
                label_w = tf.reshape(label_w, [-1, out_ch, 1])
        else:
            label_w = tf.gather(tf.transpose(w), cam_ind)
            label_w = tf.reshape(label_w, [-1, out_ch, 1])

        conv_resize = tf.reshape(conv_resize, [-1, im_width * im_width, out_ch])
        classmap = tf.matmul(conv_resize, label_w, name='classmap')
        classmap = tf.reshape(classmap, [-1, im_width, im_width], name='classmap_reshape')
        return classmap

    def count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    """
    다른 버전의 batch norm
    @classmethod
    def batch_norm(_input, is_training):
        output = tf.contrib.layers.batch_norm(_input, scale=True, \
                                              is_training=is_training, updates_collections=None)
        return output
    """
    @classmethod
    def algorithm(cls, logits):
        """
        :param y_conv: logits
        :param y_: labels
        :param learning_rate: learning rate
        :return:  pred,pred_cls , cost , correct_pred ,accuracy
        """

        print "############################################################"
        print "#                     Optimizer                            #"
        print "############################################################"
        print 'optimizer option : sgd | adam | momentum | '
        print 'selected optimizer : ', cls.optimizer_name
        print 'logits tensor Shape : {}'.format(logits.get_shape())
        print 'Preds tensor Shape : {}'.format(cls.y_.get_shape())
        print 'Learning Rate initial Value : {}'.format(cls.init_lr)
        print 'Learning Decay: {}'.format(cls.lr_decay_step)
        print '# max_iter : {}'.format(cls.max_iter)
        print 'L2 Weight Decay : {} '.format(cls.l2_weight_decay)


        optimizer_dic = {'sgd': tf.train.GradientDescentOptimizer, 'adam': tf.train.AdamOptimizer,
                         'momentum': tf.train.MomentumOptimizer}

        cls.pred_op = tf.nn.softmax(logits, name='softmax')
        cls.pred_cls_op = tf.argmax(cls.pred_op, axis=1, name='pred_cls')
        cls.cost_op= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=cls.y_), name='cost')
        cls.lr_op = tf.train.exponential_decay(cls.init_lr, cls.global_step, decay_steps=int(cls.max_iter / cls.lr_decay_step),
                                               decay_rate=0.96,
                                               staircase=False)
        # L2 Loss
        if not cls.l2_weight_decay is 0.0:
            print 'L2 Loss is Applied'
            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')
            total_cost=cls.cost_op + l2_loss * cls.l2_weight_decay
        else:
            print 'L2 Loss is Not Applied'
            total_cost = cls.cost_op
        # Select Optimizer
        if cls.optimizer_name == 'momentum':
            cls.train_op = optimizer_dic[cls.optimizer_name](cls.lr_op, use_nesterov=True).minimize(total_cost,
                                                                                                    name='train_op')
        else:
            cls.train_op = optimizer_dic[cls.optimizer_name](cls.lr_op).minimize(total_cost,name='train_op')
        # Prediction Op , Accuracy Op
        cls.correct_pred_op = tf.equal(tf.argmax(logits, 1), tf.argmax(cls.y_, 1), name='correct_pred')
        cls.accuracy_op = tf.reduce_mean(tf.cast(cls.correct_pred_op, dtype=tf.float32), name='accuracy')

    @classmethod
    def _define_input(cls, shape):
        cls.x_= tf.placeholder(tf.float32, shape=shape, name='x_')
        cls.y_ = tf.placeholder(tf.float32, shape=[None, cls.n_classes], name='y_')
        cls.cam_ind = tf.placeholder(tf.int32, shape=[], name='cam_ind')
        cls.lr_ = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        cls.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        cls.global_step = tf.placeholder(tf.int32, shape=[], name='global_step')

    @classmethod
    def sess_start(cls):

        cls.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        cls.sess.run(init)
        cls.coord = tf.train.Coordinator()
        cls.threads = tf.train.start_queue_runners(sess=cls.sess , coord = cls.coord)
    @classmethod
    def sess_stop(cls):
        cls.coord.request_stop()
        cls.coord.join(cls.threads)
        cls.sess.close()
    @classmethod
    def initialize(cls, optimizer_name, use_BN, l2_weight_decay ,logit_type, datatype, batch_size, num_epoch,
                   init_lr, lr_decay_step):

        cls.optimizer_name = optimizer_name
        cls.use_BN = use_BN

        cls.logit_type = logit_type
        cls.num_epoch = num_epoch

        cls.l2_weight_decay = l2_weight_decay
        cls.init_lr = init_lr
        cls.lr_decay_step = lr_decay_step
        ## input pipeline
        # why cls? dataprovider was used in *Train , *Test class
        cls.dataprovider = Dataprovider(datatype, batch_size, num_epoch)
        cls.max_iter =cls.dataprovider.n_train * num_epoch
        cls.n_classes = cls.dataprovider.n_classes
        cls._define_input(shape=[None, cls.dataprovider.img_h, cls.dataprovider.img_w, cls.dataprovider.img_ch])#

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
    dnn.initialize('sgd',True , True , logit_type='fc' , datatype='cifar10')
    #cls._algorithm(cls.optimizer_name)
