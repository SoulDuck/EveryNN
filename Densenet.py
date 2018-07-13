#-*- coding:utf-8 -*-
from DNN import DNN
from aug import aug_lv0 , apply_aug_lv0
import tensorflow as tf
class Densenet(DNN):
    def __init__(self, optimizer_name, use_bn, l2_weight_decay, logit_type, datatype, batch_size, cropped_size,
                 num_epoch,init_lr, lr_decay_step, model, aug_list):
        """
        :param optimizer_name:
        :param use_bn:
        :param l2_weight_decay:
        :param logit_type:
        :param datatype:
        :param batch_size:
        :param cropped_size:
        :param num_epoch:
        :param init_lr:
        :param lr_decay_step:
        :param model:
        :param aug_list:
        """
        DNN.initialize(optimizer_name, use_bn, l2_weight_decay, logit_type, datatype, batch_size, num_epoch,
                       init_lr, lr_decay_step)
        self.model = model
        self.aug_list = aug_list
        # Augmentation
        self.input = self.x_
        #  The augmentation order must be fellowed aug_rotate => aug_lv0
        if 'aug_lv0' in self.aug_list :
            print 'Augmentation Level 0 is applied'
            self.input = apply_aug_lv0(self.input,  aug_lv0 ,  self.is_training , cropped_size , cropped_size)
        # Build Model
        self.logits = self.build_graph()

        DNN.algorithm(self.logits)
        DNN.sess_start()
        self.count_trainable_params()


    def build_graph(self):

        self.growth_rate = 12 # 12 densenet 의 모델이 얼마나 향상 될지 보여줍니다
        self.depth = 40 # 100, 190, 250
        #self.stride_per_box = [2, 2, 2, 2]
        self.first_output_features = self.growth_rate * 2
        self.total_blocks = 7 # 몇번을 수축해야 할 지 결정하는 단계이다
        self.depth = 40 # 40 , 100 , 190 , 250
        self.layers_per_block = (self.depth - (self.total_blocks + 1)) // self.total_blocks
        print 'Layer per Block : {}'.format(self.layers_per_block )
        self.conv_keep_prob = 0.8



        if self.model == 'densenet_121':
            self.bottlenect_factor = 1
            self.n_blocks_per_box = [2, 2, 2, 2]
            self.bc_mode = False
        elif self.model == 'densenet_169':
            self.bottlenect_factor = 1
            self.n_blocks_per_box = [3, 4, 6, 3]
            self.bc_mode = False
        elif self.model == 'densenet_201':
            self.bottlenect_factor = 4
            self.n_blocks_per_box = [3, 4, 6, 3]
            self.bc_mode = True
        elif self.model == 'DenseNet_264':
            self.bottlenect_factor = 4
            self.n_blocks_per_box = [3, 4, 23, 3]
            self.bc_mode = True
        else:
            raise NotImplementedError

        growth_rate = self.growth_rate
        layers_per_block=self.layers_per_block #여기에 왜 있는지 모르겠는뎀 ;;;;
        with tf.variable_scope("Initial_convolution"):
            output=self.convolution2d('init_conv' ,self.x_, out_ch=self.first_output_features , k=3 , s=2)
        for i,block in enumerate(range(self.total_blocks)):
            with tf.variable_scope("Block_%d"%block):
                print '******************** BLOCK {} ***********************'.format(i)
                output=self._add_block(output,growth_rate ,layers_per_block)
                out_ch = int(output.get_shape()[3])
                output = self.convolution2d('transition_layer', output, out_ch , k=1, s=1)
                output = self.max_pool('max_pool' , output , 2, 2)

            if block != self.total_blocks -1 :
                self.top_conv = tf.identity(output, name='top_conv')


        # Get Logit
        if self.logit_type == 'gap':
            layer = self.gap(self.top_conv)
        elif self.logit_type == 'fc':
            fc_features = [4096, 4096]
            before_act_bn_mode = [False, False]
            after_act_bn_mode = [True, True]
            layer = self.top_conv
            for i in range(len(fc_features)):
                with tf.variable_scope('fc_{}'.format(str(i))) as scope:
                    print i
                    if before_act_bn_mode[i]:
                        layer = self.batch_norm_layer(layer, self.is_training, 'bn')
                    layer = self.affine(name=None, x=layer, out_ch=fc_features[i], keep_prob=0.5,
                                        is_training=self.is_training)
                    if after_act_bn_mode[i]:
                        layer = self.batch_norm_layer(layer, self.is_training, 'bn')
        else:
            print 'only ["fc", "gap"]'
            raise AssertionError
        logits = self.fc_layer_to_clssses(layer, self.n_classes)
        return logits

    def _composite_function(self , _input , out_features , kernel_size =3 ):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm_layer( _input , phase_train=self.is_training , scope_bn='BN')
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.convolution2d('conv' ,output, out_ch=out_features, k=kernel_size , s=1)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output , is_training=self.is_training , keep_prob=self.conv_keep_prob)
        return output


    def _bottlenect(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm_layer(_input , self.is_training , scope_bn='BN')
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.convolution2d('conv' , output, out_ch=inter_features, k=1, s=1 , padding='VALID')
            output = self.dropout(output, is_training=self.is_training , keep_prob=self.conv_keep_prob)
        return output

    def _add_internal_layer(self , _input , growth_rate):
        if not self.bc_mode: #Bottlenect
            comp_out= self._composite_function(_input ,out_features=growth_rate  , kernel_size=3)
        elif self.bc_mode:
            bottlenect_out = self._bottlenect(_input , out_features = growth_rate)
            comp_out = self._composite_function(bottlenect_out , out_features=growth_rate ,kernel_size=3)
        else:
            raise AssertionError

        if tf.VERSION >= 1.0:
            output= tf.concat(axis=3 , values=(_input , comp_out))
        else:
            output  = tf.concat(3 ,(_input , comp_out))
        return output

    def _add_block(self , _input , growth_rate  , layers_per_block):
        output = _input
        print _input
        for layer in range(layers_per_block):
            with  tf.variable_scope("layer_%d"%layer):
                output=self._add_internal_layer(output , growth_rate)
        output_shape = tf.shape(output)
        return output



if __name__ == '__main__':
    optimizer_name = 'sgd'
    use_bn = True
    l2_weight_decay = 0.0001
    logit_type = 'gap'
    datatype = 'cifar_10'
    batch_size = 60
    cropped_size = 512
    num_epoch = 120
    init_lr = 0.0001
    lr_decay_step = 1
    model = 'densenet_201'
    aug_list = 'aug_lv0'
    model=Densenet(optimizer_name ,use_bn , l2_weight_decay ,logit_type , datatype ,batch_size , cropped_size ,
                   num_epoch , init_lr , lr_decay_step , model , aug_list)

