#-*- coding:utf-8 -*-
import tensorflow as tf
from DNN import DNN
from aug import aug_lv0 , apply_aug_lv0 , apply_aug_rotate , tf_random_rotate_90
class RESNET_V1(DNN):
    def __init__(self, optimizer_name, use_bn, l2_weight_decay, logit_type, loss_type ,datatype, batch_size, cropped_size, num_epoch,
                       init_lr, lr_decay_step , model , aug_list):
        """
        :param n_filters_per_box: [32, 64, 64, 128 , 256 ]  , type = list
        :param n_blocks_per_box:  [3, 5 , 4, 3, 2 ]  , type = list
        :param stride_per_box: [2, 2, 2, 2 , 2 ]  , type = list
        :param use_bottlenect: True , dtype = boolean
        :param activation:  , e.g_) relu
        :param logit_type: 'gap' or 'fc' , dtype = str
        :param bottlenect_factor = 32->32-> 32*4 -> 32 필터의 수를 bottlenect 하게 합니다.

        customizing 을 함수를 추가한다.
        n_filters_per_box , n_blocks_per_box  , stride_per_box , bottlenect_factor =4
        """
        DNN.initialize(optimizer_name, use_bn, l2_weight_decay, logit_type, loss_type ,datatype, batch_size, num_epoch,
                       init_lr, lr_decay_step)

        ### bottlenect setting  ###
        """
        building model
        """
        self.model = model
        self.aug_list = aug_list
        # Augmentation
        self.input = self.x_
        #  The augmentation order must be fellowed aug_rotate => aug_lv0
        if 'aug_lv0' in self.aug_list :
            print 'Augmentation Level 0 is applied'
            self.input = apply_aug_lv0(self.input,  aug_lv0 ,  self.is_training , cropped_size , cropped_size)
        #if 'aug_rotate' in self.aug_list:
        #    self.input=tf_random_rotate_90(self.input)
        # Build Model
        self.logits = self.build_graph()
        DNN.algorithm(self.logits , self.loss_type)  # 이걸 self 로 바꾸면 안된다.
        DNN.sess_start()
        self.count_trainable_params()

    def build_graph(self):
        self.n_filters_per_box = [64, 128, 256, 512]
        self.stride_per_box = [2, 2, 2, 2]
        if self.model == 'resnet_18':
            self.bottlenect_factor=1
            self.n_blocks_per_box = [2, 2, 2, 2]
        elif self.model == 'resnet_34':
            self.bottlenect_factor = 1
            self.n_blocks_per_box = [3, 4, 6, 3]
        elif self.model == 'resnet_50':
            self.bottlenect_factor = 4
            self.n_blocks_per_box = [3, 4, 6, 3]
        elif self.model == 'resnet_101':
            self.bottlenect_factor = 4
            self.n_blocks_per_box = [3, 4, 23, 3]
        elif self.model == 'resnet_152':
            self.bottlenect_factor = 4
            self.n_blocks_per_box = [3, 8, 36, 3]
        elif self.model == 'resnet_cifar':
            self.bottlenect_factor = 1
            self.n_filters_per_box = [64, 128, 256]
            self.stride_per_box = [2, 2, 2]
            self.n_blocks_per_box = [2, 2, 2]

        else:
            raise AssertionError


        assert len(self.n_filters_per_box) == len(self.n_blocks_per_box) == len(self.stride_per_box)

        with tf.variable_scope('stem'):
            # conv filters out = 64
            layer = self.convolution2d('conv_0', out_ch= 32,  x=self.input, k=7, s=2)
            layer = self.batch_norm_layer(layer, phase_train = self.is_training, scope_bn='bn')
            #layer = self.activation(layer)
            # BN을 activation 후에 하는게 좋은지 앞에서 하는게 좋은지는 토론중이다. 난 개인적으로 weight 을 앞에다 하는게 성능을 높일거라 생각한다.
        for box_idx in range(len(self.n_filters_per_box)):
            print '#######   box_{}  ########'.format(box_idx)
            with tf.variable_scope('box_{}'.format(box_idx)):

                layer=self._box(layer , n_block= self.n_blocks_per_box[box_idx] , block_out_ch= self.n_filters_per_box[box_idx] ,
                          block_stride = self.stride_per_box[box_idx])
        self.top_conv=tf.identity(layer  , 'top_conv')

        if self.logit_type == 'gap':
            layer = self.gap(self.top_conv)
        elif self.logit_type == 'fc':
            fc_features = [4096, 4096]
            before_act_bn_mode = [False, False]
            after_act_bn_mode = [False, False]
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

    def _box(self, x,n_block , block_out_ch , block_stride):
        """
        :param x:
        :param n_block: 5  , dtype = int
        :param block_out_ch: 32 , dtype = int
        :param block_stride: 2 , dtype = int
        :return:
        """
        layer=x
        for idx in range(n_block):
            if idx == n_block-1:
                layer = self._block(layer, block_out_ch=block_out_ch, block_stride=block_stride, block_n=idx)
                #box의 마지막 block에서는 이미지를 줄이기 위해서 주어진 strides 을 convolution 에 적용시킨다
            else:
                layer = self._block(layer , block_out_ch=block_out_ch , block_stride = 1 , block_n=idx)
        return layer
    def _block(self , x , block_out_ch  , block_stride  , block_n):
        shortcut_layer = x
        layer=x
        out_ch = self.bottlenect_factor * block_out_ch

        if self.bottlenect_factor > 1: #bottlenect layer
            with tf.variable_scope('bottlenect_{}'.format(block_n)):
                layer = self.batch_norm_layer(layer , self.is_training  , 'bn_0')
                layer = self.convolution2d('conv_0' , layer , out_ch = block_out_ch , k =1 , s =1 ) #fixed padding padding = "SAME"
                layer = self.batch_norm_layer(layer, self.is_training, 'bn_1')
                layer = self.convolution2d('conv_1', layer, out_ch=block_out_ch, k=3,s=block_stride)  # fixed padding padding = "SAME"
                layer = self.batch_norm_layer(layer, self.is_training, 'bn_2')
                layer = self.convolution2d('conv_2', layer, out_ch=out_ch, k=1, s=1)  # fixed padding padding = "SAME"
                shortcut_layer = self.convolution2d('shortcut_layer', shortcut_layer, out_ch=out_ch, k=1, s=block_stride)
        elif self.bottlenect_factor == 1: #redisual layer
            with tf.variable_scope('residual_{}.'.format(block_n)):
                layer = self.convolution2d('conv_0' , layer , block_out_ch , k=3 , s=block_stride) # in here , if not block_stride = 1 , decrease image size
                layer = self.batch_norm_layer(layer , self.is_training,'bn_0' )
                layer = self.convolution2d('conv_1', layer, block_out_ch, k=3, s=1)
                shortcut_layer = self.convolution2d('shortcut_layer', shortcut_layer, out_ch=out_ch, k=1, s=block_stride)
        else:
            raise AssertionError
        return shortcut_layer + layer


class wide_resnet(object):
    def __init__(self):
        pass;


if __name__ =='__main__':
    pass;
