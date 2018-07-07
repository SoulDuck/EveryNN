#-*- coding:utf-8 -*-

import os
from DNN import DNN
import tensorflow as tf
# Missing Functions
"""


"""
class Recorder(DNN):
    def __init__(self ,folder_name):
        print '####################################################'
        print '#                 Recorder                         #'
        print '####################################################'
        self.folder_name = folder_name
        self.saver = tf.train.Saver(max_to_keep=10000000)
        self._make_logfolder()
        self._make_modelfolder()
        self.summary_writer = tf.summary.FileWriter(self.logs_path)
        self.summary_writer.add_graph(tf.get_default_graph())


    def write_lr(self   , learning_rate , step):
        summary = tf.Summary(value=[tf.Summary.Value(tag='learning_rate' , simple_value = float(learning_rate))])
        self.summary_writer.add_summary(summary , step)

    def write_acc_loss(self , prefix, loss, acc, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag='loss_{}'.format(prefix), simple_value=float(loss)),
                                    tf.Summary.Value(tag='accuracy_{}'.format(prefix), simple_value=float(acc))])
        self.summary_writer.add_summary(summary, step)

    def cwrite_spec_sens(self , prefix, spec, sens ,step):
        summary = tf.Summary(value=[tf.Summary.Value(tag='spec {}'.format(prefix), simple_value=float(spec)),
                                    tf.Summary.Value(tag='sens {}'.format(prefix), simple_value=float(sens))])
        self.summary_writer.add_summary(summary,step)


    def _make_logfolder(self):
        log_count =0;
        while True:
            logs_root_path='./logs/{}'.format(self.folder_name)
            try:
                os.makedirs(logs_root_path)
            except Exception as e :
                pass;
            self.logs_path=os.path.join( logs_root_path , str(log_count))
            if not os.path.isdir(self.logs_path):
                os.mkdir(self.logs_path)
                break;
            else:
                log_count+=1
        print 'folder where logs is saved : {} '.format(self.logs_path)



    def _make_modelfolder(self):
        model_count =0;
        while True:
            models_root_path='./models/{}'.format(self.folder_name)
            try:
                os.makedirs(models_root_path)
            except Exception as e:
                pass;
            self.models_path=os.path.join(models_root_path , str(model_count))

            if not os.path.isdir(self.models_path):
                os.mkdir(self.models_path)
                break; #
            else:
                model_count+=1

        print 'folder where models is saved : {} '.format(self.models_path)