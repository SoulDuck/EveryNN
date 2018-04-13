#-*- coding:utf-8 -*-
from DNN import DNN
import numpy as np
import tensorflow as tf
class Tester(DNN):

    def __init__(self , recorder):
        print '####################################################'
        print '#                   Tester                         #'
        print '####################################################'
        self.recorder = recorder


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
        self.is_training_ = tf.get_default_graph().get_tensor_by_name('is_training:0')
        self.top_conv = tf.get_default_graph().get_tensor_by_name('top_conv:0')
        self.logits_ = tf.get_default_graph().get_tensor_by_name('logits:0')
        self.cam_ = tf.get_default_graph().get_tensor_by_name('classmap:0')
        try:
            cam_ind = tf.get_default_graph().get_tensor_by_name('cam_ind:0')
        except Exception as e :
            print "CAM 이 구현되어 있지 않은 모델입니다."
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
        self.recorder.write_acc_loss(prefix='Test' , loss=mean_loss , acc= mean_loss , step= step)

        return mean_acc, mean_loss, pred_all
