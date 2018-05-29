#-*- coding:utf-8 -*-
from DNN import DNN
import tensorflow as tf
from aug import aug_lv0
#ef convolution2d(name,x,out_ch,k=3 , s=2 , padding='SAME'):
class INCEPTION_V4(DNN):
    def __init__(self, optimizer_name, use_bn, l2_weight_decay, logit_type, datatype, batch_size, croppped_size, num_epoch,
                       init_lr, lr_decay_step , model , aug_level):
        DNN.initialize(optimizer_name, use_bn, l2_weight_decay, logit_type, datatype, batch_size, num_epoch,
                       init_lr, lr_decay_step)

        self.model = model
        self.aug_level = aug_level
        # Augmentation
        if self.aug_level == 'aug_lv0' :
            self.input = aug_lv0(self.x_ , self.is_training ,(croppped_size , croppped_size))
        else:
            self.input = self.x_

        # Build Model
        self.logits = self.build_graph()

        DNN.algorithm(self.logits)  # 이걸 self 로 바꾸면 안된다.
        self.count_trainable_params()
        DNN.sess_start()


    def build_graph(self):
        if self.model == 'A':
            self.top_conv = self.structure_A(self.input)
        elif self.model == 'B':
            self.top_conv = self.structure_B(self.input)
        else:
            raise AssertionError

        if self.logit_type == 'gap':
            layer = self.gap(self.top_conv)
        elif self.logit_type == 'fc':
            fc_features = [4096, 4096]
            before_act_bn_mode = [False, False]
            after_act_bn_mode = [False, False]
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
            print '["fc", "gap"]'
            raise AssertionError
        self.logits = self.fc_layer_to_clssses(layer, self.n_classes)
        return self.logits




    def stem(self,name , x):
        with tf.variable_scope(name) as scope:
            layer=self.convolution2d('cnn_0',x,32,k=3,s=2 , padding='VALID')
            layer = self.convolution2d('cnn_1',layer, 32, k = 3, s = 1, padding = 'VALID')
            layer = self.convolution2d('cnn_2', layer, 64, k=3, s=1, padding='SAME')
            layer_1 = self.max_pool('max_3', layer, k=3, s=2, padding='VALID')
            layer_2 = self.convolution2d('cnn_3_1', layer, 96, k=3, s=2, padding='VALID')
            layer_join=tf.concat([layer_1,layer_2] , axis=3 , name='join')
            print 'layer_name :','join'
            print 'layer_shape :',layer_join.get_shape()
        return layer_join
    def stem_1(self,name , x ):
        with tf.variable_scope(name) as scope:
            layer = self.convolution2d('cnn_0', x, 64, k=1, s=1)
            layer = self.convolution2d('cnn_1', layer, 96, k=3, s=1, padding='VALID')
            layer_ = self.convolution2d('cnn__0', x, 64, k=1, s=1)
            layer_ = self.convolution2d_manual('cnn__1', layer_, 64, k_h=7,k_w=1, s=1)
            layer_ = self.convolution2d_manual('cnn__2', layer_, 64, k_h=1,k_w=7,s=1 )
            layer_ = self.convolution2d('cnn__3', layer_, 96, k=3, s=1, padding='VALID')

            layer_join = tf.concat([layer, layer_], axis=3, name='join')
            print 'layer_name :','join'
            print 'layer_shape :',layer_join.get_shape()
        return layer_join
    def stem_2(self,name ,x ):
        with tf.variable_scope(name) as scope:
            layer= self.convolution2d('cnn_0' , x, 192,k=3,s=2,padding='VALID')
            layer_=self.max_pool('max__0' , x, k=3 , s=2 , padding = 'VALID')
            layer_join = tf.concat([layer , layer_] , axis = 3 ,name='join')
            print 'layer_name :','join'
            print 'layer_shape :',layer_join.get_shape()
        return layer_join

    def reductionA(self,name,x ):

        with tf.variable_scope(name) as scope:
            layer_ =self.max_pool('max_pool_0' ,x, k=3, s=2 ,padding='VALID')
            layer__ =self.convolution2d('cnn__0' ,x,192 , k=3 , s=2 , padding='VALID'  )
            layer___ = self.convolution2d('cnn___0',x,224, k=1, s=1, padding='SAME')
            layer___ = self.convolution2d('cnn___1',layer___,256, k=3, s=1, padding='SAME')
            layer___ = self.convolution2d('cnn___2',layer___,385, k=3, s=2, padding='VALID')

            layer_join=tf.concat([layer_ , layer__ , layer___ ], axis=3 , name='join')
            print 'layer_name :','join'
            print 'layer_shape :',layer_join.get_shape()
        return layer_join

    def reductionB(self,name , x):
        with tf.variable_scope(name) as scope:
            layer_ = self.max_pool('self.max_pool_0',x, k=3, s=2, padding='VALID')

            layer__ = self.convolution2d('cnn__0',x,192, k=1, s=1, padding='SAME')
            layer__ = self.convolution2d('cnn__1' ,layer__, 192,k=3 ,s=2 ,padding='VALID')

            layer___ = self.convolution2d('cnn___0',x,256, k=1, s=1, padding='SAME')
            layer___ = self.convolution2d_manual('cnn___1',layer___,256, k_h=1 , k_w=7, s=1, padding='SAME')
            layer___ = self.convolution2d_manual('cnn___2',layer___,320, k_h=7, k_w=1, s=1, padding='SAME')
            layer___ = self.convolution2d('cnn___3',layer___, 320,k=3, s=2, padding='VALID')

            layer_join=tf.concat([layer_ , layer__ , layer___] , axis=3 , name='join')
            print 'layer_name :','join'
            print 'layer_shape :',layer_join.get_shape()
        return layer_join

    def reductionC(self,name ,x ):
        with tf.variable_scope(name) as scope:
            layer=self.max_pool('max_pool0' , x ,3,2 , padding='VALID')

            layer_=self.convolution2d('cnn_0',x , 256,1,1)
            layer_=self.convolution2d('cnn_1',layer_, 384 , 3,1 , padding='VALID')

            layer__ = self.convolution2d('cnn_0', x, 256, 1, 1)
            layer__ = self.convolution2d('cnn_1', layer__, 256, 3, 1 , padding='VALID')

            layer___= self.convolution2d('cnn__0',x, 256 , 1,1)
            layer___= self.convolution2d('cnn__1',layer___,256 , 3,1)
            layer___= self.convolution2d('cnn__2',layer___,256 , 3,1 , padding='VALID')

            layer_join= tf.concat([layer , layer_ , layer__ , layer___], axis=3 , name='join' )
            return layer_join



    def blockB(self,name , x):
        with tf.variable_scope(name) as scope:
            layer=self.avg_pool('avg_pool', x, k=2 , s=1)
            layer=self.convolution2d('cnn',layer,128,k=1,s=1)

            layer_=self.convolution2d('cnn_0',x,384,k=1,s=1)

            layer__=self.convolution2d('cnn__0',x,192,k=1,s=1)
            layer__ = self.convolution2d_manual('cnn__1', layer__, 224, k_h=1 , k_w=7, s=1 )
            layer__ = self.convolution2d_manual('cnn__2', layer__, 256, k_h=1 , k_w=7, s=1 )

            layer___=self.convolution2d('cnn___0',x,192,k=1,s=1)
            layer___=self.convolution2d_manual('cnn___1',layer___,192,k_h=1,k_w=7,s=1)
            layer___=self.convolution2d_manual('cnn___2',layer___,224,k_h=7,k_w=1,s=1)
            layer___=self.convolution2d_manual('cnn___3',layer___,224,k_h=1,k_w=7,s=1)
            layer___=self.convolution2d_manual('cnn___4',layer___,256,k_h=7,k_w=1,s=1)

            layer_join=tf.concat([layer, layer_ , layer__ , layer___] , axis=3 , name ='join')
            print 'layer_name :','join'
            print 'layer_shape :',layer_join.get_shape()
        return layer_join

    def blockA(self,name , x):
        with tf.variable_scope(name) as scope:
            layer = self.avg_pool('avg_pool', x, k=2, s=1)
            layer = self.convolution2d('cnn', layer, 96, k=1, s=1)

            layer_ = self.convolution2d('cnn_0', x, 96, k=1, s=1)

            layer__ = self.convolution2d('cnn__0', x, 64, k=1, s=1)
            layer__ = self.convolution2d('cnn__1', layer__,96, k=3, s=1)

            layer___ = self.convolution2d('cnn___0', x,64, k=1, s=1)
            layer___ = self.convolution2d('cnn___1',layer___,96, k=3, s=1)
            layer___ = self.convolution2d('cnn___2',layer___,96, k=3, s=1)

            layer_join = tf.concat([layer, layer_, layer__, layer___], axis=3, name='join')
            print 'layer_name :', 'join'
            print 'layer_shape :', layer_join.get_shape()
        return layer_join

    def blockC(self,name , x):
        with tf.variable_scope(name) as scope:
            layer = self.avg_pool('avg_pool', x, k=2, s=1)
            layer = self.convolution2d('cnn', layer, 256, k=1, s=1)

            layer_ = self.convolution2d('cnn_0', x, 256, k=1, s=1)

            layer__ = self.convolution2d('cnn__0',x, 384, k=1, s=1)
            layer__0 = self.convolution2d_manual('cnn__1_0',layer__ , 256, k_h=1,k_w=3, s=1)
            layer__1 = self.convolution2d_manual('cnn__1_1',layer__ , 256, k_h=3,k_w=1, s=1)

            layer___ = self.convolution2d('cnn___0', x,384 ,k=1, s=1)
            layer___ = self.convolution2d_manual('cnn___1', layer___,448, k_h=1, k_w=3 ,s=1)
            layer___ = self.convolution2d_manual('cnn___2', layer___,512, k_h=3 , k_w=1, s=1)
            layer___0 = self.convolution2d_manual('cnn___3_0', layer___, 256, k_h=3,k_w=1, s=1)
            layer___1 = self.convolution2d_manual('cnn___3_1', layer___,256, k_h=1,k_w=3, s=1)
            layer_join = tf.concat([layer, layer_, layer__0, layer__1 ,layer___0 , layer___1], axis=3, name='join')
            print 'layer_name :', 'join'
            print 'layer_shape :', layer_join.get_shape()
            return layer_join

    def resnet_blockA(self,name ,x):
        with tf.variable_scope(name) as scope:
            layer=self.convolution2d('cnn0',x,128,1,1)
            layer_ = self.convolution2d('cnn_0', x, 128, 1, 1)
            layer_ = self.convolution2d_manual('cnn_1', layer_, 128, k_h=1,k_w=7 ,s=1)
            layer_ = self.convolution2d_manual('cnn_2', layer_, 128, k_h=7,k_w=1 ,s=1)
            layer_join=tf.concat([layer , layer_], axis=3 , name='join')
            layer_join=self.convolution2d('layer_join_cnn' , layer_join , 897 , 1,1)
            print 'layer_name :', 'join'
            print 'layer_shape :', layer_join.get_shape()

            if x.get_shape()[-1] != layer_join.get_shape()[-1]:
                x=self.convolution2d('upscale_dimension',x, layer_join.get_shape()[-1] , k=1,s=1)
            layer_join=tf.add(x,layer_join , 'add')
            return layer_join
    def resnet_blockB(self,name , x):
        with tf.variable_scope(name) as scope:

            layer = self.convolution2d('cnn0', x, 32, 1, 1)
            layer_ = self.convolution2d('cnn_0', x, 32, 1, 1)
            layer_ = self.convolution2d('cnn_1', layer_, 32, 3, 1)
            layer__ = self.convolution2d('cnn__0', x, 32, 1, 1)
            layer__ = self.convolution2d('cnn__1', layer__, 32, 3, 1)
            layer__ = self.convolution2d('cnn__2', layer__, 32, 3, 1)

            layer_join=tf.concat([layer , layer_ , layer__] , axis=3 , name='join')
            layer_join=self.convolution2d('layer_join_cnn' , layer_join ,256, 1,1 )

            if x.get_shape()[-1] != layer_join.get_shape()[-1]:
                x=self.convolution2d('upscale_dimension',x, layer_join.get_shape()[-1] , k=1,s=1)
            layer_join=tf.add(x,layer_join )
            return layer_join
    def resnet_blockC(self,name, x):
        with tf.variable_scope(name) as scope:
            layer = self.convolution2d('cnn0', x, 192, 1, 1)
            layer_ = self.convolution2d('cnn_0', x, 192, 1, 1)
            layer_ = self.convolution2d_manual('cnn_1', layer_, 192, k_h=1, k_w=3, s=1)
            layer_ = self.convolution2d_manual('cnn_2', layer_, 192, k_h=3, k_w=1, s=1)
            layer_join = tf.concat([layer, layer_], axis=3, name='join')
            layer_join = self.convolution2d('layer_join_cnn', layer_join, 1792, 1, 1)
            if x.get_shape()[-1] != layer_join.get_shape()[-1]:
                x=self.convolution2d('upscale_dimension',x, layer_join.get_shape()[-1] , k=1,s=1)
            layer_join = tf.add(x, layer_join, 'add')
            print 'layer_name :', 'join'
            print 'layer_shape :', layer_join.get_shape()
            return layer_join

    def structure_A(self,x_):
        print 'stem A -> B -> C -> blockA -> reductionA -> blockB -> reduction B -> blockC'
        layer = self.stem('stem', x_)
        layer = self.stem_1('stem_1', layer)
        layer = self.stem_2('stem_2', layer)
        layer = self.blockA('blockA_0', layer)
        layer = self.reductionA('reductionA', layer)
        layer = self.blockB('blockB_0', layer)
        layer = self.reductionB('reductionB', layer)
        layer = self.blockC('blockC_0', layer)
        top_conv = tf.identity(layer, name='top_conv')
        return top_conv

    def structure_B(self, x_, phase_train):
        print 'stem A -> B -> C -> blockA -> reductionA -> blockB -> reduction B -> blockC'
        layer = self.stem('stem', x_)
        layer=self.batch_norm_layer(layer,phase_train,'stem_bn')
        layer = self.stem_1('stem_1', layer)
        layer=self.batch_norm_layer(layer,phase_train,'stem1_bn')
        layer = self.stem_2('stem_2', layer)
        layer=self.batch_norm_layer(layer,phase_train,'stem2_bn')
        layer = self.blockA('blockA_0', layer)
        layer = self.reductionA('reductionA', layer)
        layer = self.blockB('blockB_0', layer)
        layer = self.batch_norm_layer(layer,phase_train,'reductionA_bn')
        layer = self.reductionB('reductionB', layer)
        layer = self.blockC('blockC_0', layer)
        layer = tf.identity(layer, name='top_conv')
        return layer





