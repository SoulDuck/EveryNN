from utils import show_progress
from DNN import DNN
import Dataprovider
import numpy as np
import utils
class Trainer(DNN):
    def __init__(self , recorder):
        print '####################################################'
        print '#                   Trainer                        #'
        print '####################################################'
        self.train_iter = 100
        self.train_step = 0
        self.recorder = recorder
        self.train_acc=0
        self.train_loss=0

    def _lr_scheduler(self , step):
        if step < 5000:
            learning_rate = 0.001
        elif step < 45000:
            learning_rate = 0.0007
        elif step < 60000:
            learning_rate = 0.0005
        elif step < 120000:
            learning_rate = 0.0001
        else:
            learning_rate = 0.00001
            ####
        return learning_rate
    def training(self):

        max_iter = self.train_step+self.train_iter
        for step in range(self.train_step, max_iter):
            show_progress(step , max_iter )
            learning_rate = self._lr_scheduler(step)
            #### learning rate schcedule
            """ #### Traininig  ### """
            train_fetches = [self.train_op, self.accuracy_op, self.cost_op]
            batch_xs , batch_ys=self.sess.run([self.dataprovider.batch_xs ,self.dataprovider.batch_ys])

            #utils.plot_images(batch_xs)
            if np.max(batch_xs) > 1:
                batch_xs=batch_xs/255.

            train_feedDict = {self.x_: batch_xs, self.y_: batch_ys, self.cam_ind: 0, self.lr_: learning_rate,
                              self.is_training: True , self.global_step : step}

            _, self.train_acc, self.train_loss = self.sess.run(fetches=train_fetches, feed_dict=train_feedDict)
            # print 'train acc : {} loss : {}'.format(train_acc, train_loss)
            self.recorder.write_acc_loss('Train' , self.train_loss , self.train_acc , self.train_step)
        self.train_step = step

