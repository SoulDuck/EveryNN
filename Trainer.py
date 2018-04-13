from utils import show_progress
from DNN import DNN
class Trainer(DNN ):
    def __init__(self , recorder):
        self.recorder = recorder


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
    def training(self, max_iter , global_step , batch_size):
        max_acc=0
        min_loss=0
        for step in range(global_step , global_step+max_iter):
            show_progress(step , max_iter)
            learning_rate = self._lr_scheduler(step)
            #### learning rate schcedule
            """ #### Traininig  ### """
            train_fetches = [self.train_op, self.accuracy_op, self.loss_op]
            batch_xs, batch_ys, batch_fname = self.pipeline.next_batch(batch_size)
            train_feedDict = {self.x_: batch_xs, self.y_: batch_ys, self.cam_ind: 0, self.lr_: learning_rate,
                              self.is_training: True}
            _, train_acc, train_loss = self.sess.run(fetches=train_fetches, feed_dict=train_feedDict)
            # print 'train acc : {} loss : {}'.format(train_acc, train_loss)
            self.recorder.write_acc_loss('Train' , train_loss , train_acc , step )




"""
            if step % ckpt == 0:
                test_fetches = [accuracy_op, loss_op, pred_op]
                val_acc, val_loss, val_preds = validate(test_fetches, val_imgs, val_labs, batch_size)  # Validate 합니다.
                max_acc = save_best_acc(val_acc, max_acc, step)
                min_loss = save_best_loss(val_loss, min_loss, step)
                write_summary(summary_writer, step, learning_rate, 'validation', val_acc, val_loss)
                # add learning rate summary
"""