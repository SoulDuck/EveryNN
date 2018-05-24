#-*- coding:utf-8 -*-
from DNN import DNN
import numpy as np
import tensorflow as tf
import sys
import utils
from PIL import Image
class Tester(DNN):

    def __init__(self , recorder):
        print '####################################################'
        print '#                   Tester                         #'
        print '####################################################'
        self.recorder = recorder
        self.val_acc=0
        self.val_loss=0
        self.max_acc=0
        self.min_loss=10000000


    def get_acc(self,trues, preds):
        assert np.ndim(trues) == np.ndim(preds), 'true shape : {} pred shape : {} '.format(np.shape(trues), np.shape(preds))
        if np.ndim(trues) == 2:
            true_cls = np.argmax(trues, axis=1)
            pred_cls = np.argmax(preds, axis=1)

        tmp = [true_cls == pred_cls]
        acc = np.sum(tmp) / float(len(true_cls))
        return acc
    def _reconstruct_model(self , model_path):
        print 'Reconstruct Model';
        sess = tf.Session()
        saver = tf.train.import_meta_graph(
            meta_graph_or_file=model_path + '.meta')  # example model path ./models/fundus_300/5/model_1.ckpt
        saver.restore(sess, save_path=model_path)  # example model path ./models/fundus_300/5/model_1.ckpt

        #Naming Rule  : tensor 뒤에는 _ undersocre 을 붙입니다.
        self.x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
        self.y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
        self.pred_ = tf.get_default_graph().get_tensor_by_name('softmax:0')
        self.is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
        self.top_conv = tf.get_default_graph().get_tensor_by_name('top_conv:0')
        self.logits_ = tf.get_default_graph().get_tensor_by_name('logits:0')
        self.cam_ = tf.get_default_graph().get_tensor_by_name('classmap:0')
        try:
            cam_ind = tf.get_default_graph().get_tensor_by_name('cam_ind:0')
        except Exception as e :
            print "CAM 이 구현되어 있지 않은 모델입니다."
    def show_acc_loss(self , step ):
        print ''
        if not step is None:
            print 'Step : {}'.format(step)
        print 'Validation Acc : {} | Loss : {}'.format(self.acc, self.loss)
        print ''

    def validate(self , imgs , labs , batch_size , step):

        """
        #### Validate ###
        test_fetches = mean_cost , pred
        """
        share = len(labs) / batch_size
        remainer= len(labs) % batch_size
        loss_all,  pred_all = [], []
        fetches= [self.cost_op , self.pred_op]
        if share !=0:
            for i in range(share):  # 여기서 테스트 셋을 sess.run()할수 있게 쪼갭니다
                feedDict = {self.x_: imgs[i * batch_size:(i + 1) * batch_size],
                                 self.y_: labs[i * batch_size:(i + 1) * batch_size], self.is_training: False}
                mean_loss, preds = self.sess.run(fetches=fetches, feed_dict=feedDict)
                pred_all.extend(preds)
                loss_all.append(mean_loss)

            if remainer != 0 and share != 0:
                feedDict = {self.x_: imgs[-remainer:], self.y_: labs[-remainer:], self.is_training: False}
                val_loss, preds = self.sess.run(fetches=fetches, feed_dict=feedDict)
                pred_all.extend(preds)
                loss_all.append(mean_loss)
        else:
            test_feedDict = {self.x_: imgs, self.y_:labs, self.is_training: False}
            loss_all, pred_all = self.sess.run(fetches=fetches, feed_dict=test_feedDict)

        mean_loss=np.mean(loss_all)
        mean_acc = self.get_acc(labs,  pred_all)
        self.recorder.write_acc_loss(prefix='Test' , loss=mean_loss , acc= mean_acc , step= step)
        self.acc = mean_acc
        self.loss = mean_loss

        if self.acc > self.max_acc:
            self.max_acc = self.acc
            print '###### Model Saved ######'
            print 'Max Acc : {}'.format(self.max_acc)
            self.recorder.saver.save(sess = DNN.sess ,save_path = self.recorder.models_path)

        if self.loss < self.min_loss:
            self.loss = self.min_loss



    def validate_tfrecords(self , tfrecord_path , preprocessing , resize):
        """
        Validate 이용해 데이터를 꺼내옵니다. generators 임으로 하나하나 씩 꺼내 옵니다.
        callback 함수로 aug 함수를 전달합니다
        :return:
        """
        loss_all, pred_all, labels = [], [], []
        record_iter = tf.python_io.tf_record_iterator(path = tfrecord_path)
        fetches = [self.cost_op, self.pred_op]
        for i , str_record in enumerate(record_iter):
            msg = '\r -progress {0}'.format(i)
            sys.stdout.write(msg)
            sys.stdout.flush()
            example = tf.train.Example()
            example.ParseFromString(str_record)
            height = int(example.features.feature['height'].int64_list.value[0])
            width = int(example.features.feature['width'].int64_list.value[0])
            raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
            label = int(example.features.feature['label'].int64_list.value[0])
            filename = (example.features.feature['filename'].bytes_list.value[0])

            # Reconstruct Image
            image = np.fromstring(raw_image, dtype=np.uint8)
            image = image.reshape((height, width, -1))
            image=np.expand_dims(image, axis=0)
            if  np.max(image) > 1:
                image=image/255.

            # Resize
            if not resize is None:
                image = np.asarray(Image.fromarray(image).resize(resize, Image.ANTIALIAS))
            # Preprocessing
            if not preprocessing is None:
                image = preprocessing(image)
            # CLS ==> One Hot encoding
            label=utils.cls2onehot([label] ,self.n_classes)
            labels.extend(label)

            # Run Test
            test_feedDict = {self.x_: image, self.y_: label, self.is_training: False}
            loss , pred = self.sess.run(fetches=fetches, feed_dict=test_feedDict)
            loss_all.append(loss)
            pred_all.extend(pred)
        mean_loss = np.mean(loss_all)
        mean_acc = self.get_acc(labels, pred_all)
        return mean_acc , mean_loss , pred_all
"""
    @classmethod
    def reconstruct_tfrecord_rawdata(cls, tfrecord_path, resize):
        debug_flag_lv0 = False
        debug_flag_lv1 = False
        if __debug__ == debug_flag_lv0:
            print 'debug start | batch.py | class tfrecord_batch | reconstruct_tfrecord_rawdata '

        print 'now Reconstruct Image Data please wait a second'
        print 'Resize {}'.format(resize)
        reconstruct_image = []
        # caution record_iter is generator
        record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)

        ret_img_list = []
        ret_lab_list = []
        ret_filename_list = []
        for i, str_record in enumerate(record_iter):
            msg = '\r -progress {0}'.format(i)
            sys.stdout.write(msg)
            sys.stdout.flush()
            example = tf.train.Example()
            example.ParseFromString(str_record)

            height = int(example.features.feature['height'].int64_list.value[0])
            width = int(example.features.feature['width'].int64_list.value[0])
            raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
            label = int(example.features.feature['label'].int64_list.value[0])
            filename = (example.features.feature['filename'].bytes_list.value[0])
            
            image = np.fromstring(raw_image, dtype=np.uint8)
            image = image.reshape((height, width, -1))
            if not resize is None:
                image=np.asarray(Image.fromarray(image).resize(resize,Image.ANTIALIAS))
            ret_img_list.append(image)
            ret_lab_list.append(label)
            ret_filename_list.append(filename)


        ret_img = np.asarray(ret_img_list)
        ret_lab = np.asarray(ret_lab_list)
        if debug_flag_lv1 == True:
            print ''
            print 'images shape : ', np.shape(ret_img)
            print 'labels shape : ', np.shape(ret_lab)
            print 'length of filenames : ', len(ret_filename_list)
        return ret_img, ret_lab, ret_filename_list
"""

