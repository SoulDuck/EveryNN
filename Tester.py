#-*- coding:utf-8 -*-
from DNN import DNN
import numpy as np
import tensorflow as tf
import sys , os
import utils
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import time
import sklearn
class Tester(DNN):
    def __init__(self , recorder ):
        print '####################################################'
        print '#                   Tester                         #'
        print '####################################################'
        if not recorder is None:
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



    def _reconstruct_model(self , model_path , cam_imgSize=540):
        tf.reset_default_graph()
        print 'Reconstruct Model';
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(
            meta_graph_or_file=model_path + '.meta')  # example model path ./models/fundus_300/5/model_1.ckpt
        saver.restore(self.sess, save_path=model_path)  # example model path ./models/fundus_300/5/model_1.ckpt

        #Naming Rule  : tensor 뒤에는 _ undersocre 을 붙입니다.
        self.x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
        self.y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
        self.pred_op = tf.get_default_graph().get_tensor_by_name('softmax:0')
        self.cost_op = tf.get_default_graph().get_tensor_by_name('cost:0')
        self.is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
        self.top_conv = tf.get_default_graph().get_tensor_by_name('top_conv:0')
        self.logits_ = tf.get_default_graph().get_tensor_by_name('logits:0')
        self.final_w= tf.get_default_graph().get_tensor_by_name('final/w:0')
        try:
            cam_ind = tf.get_default_graph().get_tensor_by_name('cam_ind:0')
        except Exception as e :
            print "CAM 이 구현되어 있지 않은 모델입니다."
        self.classmap_op = self.get_class_map('final', self.top_conv, 1, cam_imgSize , self.final_w)

    def show_acc_loss(self , step ):
        print ''
        if not step is None:
            print 'Step : {}'.format(step)
        print 'Validation Acc : {} | Loss : {}'.format(self.acc, self.loss)
        print 'Max Valication Acc : {} | Min Loss {}'.format(self.max_acc, self.min_loss)
        print ''

    def show_acc_by_label(self):
        for ind_cls in range(self.n_classes):
            print 'Label : {} , Accuracy : {} '.format(ind_cls , self.acc_by_labels[ind_cls])

    def validate(self , imgs , labs , batch_size , step , save_model = True):
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

        self.pred_all = pred_all
        self.loss = np.mean(loss_all)
        self.acc = self.get_acc(labs,  self.pred_all)


        # Accuracy By Label
        self.acc_by_labels=[]
        cls = np.argmax(labs, axis=1)
        for cls_ind in range(self.n_classes):
            indices = [cls == cls_ind]
            lab_by_true = labs[indices]
            np.sum(lab_by_true)
            lab_by_pred = np.asarray(self.pred_all)[indices]
            lab_by_acc = self.get_acc(lab_by_true, lab_by_pred)
            self.acc_by_labels.append(lab_by_acc)

        # Sensitivity , Specifity
        self.get_spec_sens()



        if save_model:
            self.recorder.write_acc_loss(prefix='Test', loss=self.loss, acc=self.acc, step=step)
            if self.acc > self.max_acc:
                self.max_acc = self.acc
                print '###### Model Saved ######'
                print 'Max Acc : {}'.format(self.max_acc)
                self.recorder.saver.save(sess = DNN.sess ,save_path = os.path.join(self.recorder.models_path , 'model') , global_step = step)

            if self.loss < self.min_loss:
                self.min_loss = self.loss



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

    def validate_top_k(self , preds, labels, top_k , show_flag):
        accs = []
        top_k_acc = []
        for k in range(1, top_k + 1):
            for i, pred in enumerate(preds):
                indices = pred.argsort()[-k:][::-1]
                bin_value = np.sum(labels[i][indices])  # bin_value  True , False
                accs.append(bin_value)
            acc = np.sum(accs) / float(len(accs))
            if show_flag:
                print '{} : Accuracy {}'.format(k,acc)
            top_k_acc.append(acc)

        return top_k_acc

    def _extract_actmap(self , imgs):

        for i,img in enumerate(imgs):

            feed_dict = {self.x_: imgs[i:i+1], self.is_training: False}
            cam = self.sess.run(self.classmap_op, feed_dict=feed_dict)
            cam = np.squeeze(cam)
            cam = list(map(lambda x: (x - x.min()) / (x.max() - x.min()), cam))
            cmap = plt.cm.jet
            cam = cmap(cam)
            plt.imsave('tmp_cam.png' , cam)
            plt.imsave('tmp_img.png', imgs[i])

    def black_box(self , oriimg ,box_size):
        ret_dict= {}

        height , width=np.shape(oriimg)[:2]
        skip_pix=30
        count=0
        for h_ind in range(0,height-box_size+1,skip_pix):
            for w_ind in range(0,width-box_size+1,skip_pix):
                img = copy.deepcopy(np.asarray(oriimg))
                img[h_ind : h_ind+ box_size , w_ind : w_ind+ box_size ] = 0
                #plt.imsave('tmp_blackbox/tmp_blackbox_{}.png'.format(count) , img )
                count += 1
                ret_dict[count] = ([h_ind , w_ind , box_size] , img )
        return ret_dict

    def mask(self , img , threshold , mark , bin_flag):
        assert mark in  ['<' ,'>' , '<=' , '>='] , "mark param = '<' ,'>' , '<=' , '>=' "
        img=np.asarray(img)
        flatted_img = img.reshape(-1)
        if mark == '<':
            indices =np.where(flatted_img < threshold)[0]
        elif mark == '>':
            indices = np.where(flatted_img > threshold)[0]
            rev_indices = np.where(flatted_img >= threshold)[0]
        elif mark == '<=':
            indices = np.where(flatted_img <= threshold)[0]
        elif mark == '>=':
            indices = np.where(flatted_img >= threshold)[0]
        else:
            raise AssertionError

        # difference of sets
        rev_indices=list(set(range(len(flatted_img))) - set(indices))

        # mask imgs
        flatted_img[indices] = 255
        if bin_flag :
            # if Meet the condition =>255 , else 0
            flatted_img[rev_indices] = 0

        masked_img=flatted_img.reshape(np.shape(img))
        return masked_img


    def eval(self, model_path, test_imgs, test_labs, batch_size , actmap_dir):
        self._reconstruct_model(model_path)
        self.validate(test_imgs , test_labs , batch_size , step=0 , save_model = False)
        if not actmap_dir is None:
            cams = self._extract_actmap(test_imgs)
            print np.shape(cams)
        return self.pred_all , self.acc , self.loss

    def plotROC(self,predStrength, labels , prefix , savepath):
        debug_flag = False
        assert np.ndim(predStrength) == np.ndim(labels)
        if np.ndim(predStrength) == 2:
            predStrength = np.argmax(predStrength, axis=1)
            labels = np.argmax(labels, axis=1)

        # how to input?

        cursor = (1.0, 1.0)  # initial cursor
        ySum = 0.0  # for AUC curve
        n_pos = np.sum(np.array(labels) == 1)
        n_neg = len(labels) - n_pos
        print n_pos
        print n_neg
        y_step = 1 / float(n_pos)
        x_step = 1 / float(n_neg)
        n_est_pos = 0
        sortedIndices = np.argsort(predStrength, axis=0)
        fig = plt.figure(figsize=(10,10))
        fig.clf()
        ax = plt.subplot(1, 1, 1)
        if __debug__ == debug_flag:
            print 'labels', labels[:10]
            print 'predStrength', predStrength.T[:10]
            print 'sortedIndices', sortedIndices.T[:10]
            print  sortedIndices.tolist()[:10]
        for ind in sortedIndices.tolist():
            print ind
            if labels[ind] == 1.0:
                DelX = 0;
                DelY = y_step
            else:
                DelX = x_step;
                DelY = 0
                ySum += cursor[1]
            ax.plot([cursor[0], cursor[0] - DelX], [cursor[1], cursor[1] - DelY])
            cursor = (cursor[0] - DelX, cursor[1] - DelY)
            if __debug__ == debug_flag:
                print 'label', labels[ind]
                print 'delX',
                print 'sortedIndices', sortedIndices.T
                print 'DelX:', DelX, 'DelY:', DelY
                print 'cursor[0]-DelX :', cursor[0], 'cursor[1]-DelY :', cursor[1]
        ax.plot([0, 1], [0, 1], 'b--')
        plt.xlabel('False Positive Rate');
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for {}'.format(prefix))

        ax.axis([0, 1, 0, 1])
        if __debug__ == debug_flag:
            print '# of True :', n_pos
            print '# of False :', n_neg
        plt.savefig(savepath)
        # plt.show()
        print 'The Area Under Curve is :', ySum * x_step

    def get_confmat(self , pred_cls , cls):
        cm = sklearn.confusion_matrix(pred_cls[['active_cust']], cls)
        return cm

    def get_spec_sens(self, pred_cls , labels):
        cm=self.get_confmat(pred_cls , labels )
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        print('Sensitivity : ', sensitivity)
        print('Specificity : ', specificity)
        return sensitivity , specificity

    def ensemble(self , test_imgs , test_labs , batch_size, *model_paths):
        # Calculate predictions
        pred_dic={}
        for model_path in model_paths:
            print 'Model Path : {} is Loading : '.format(model_path)
            self._reconstruct_model(model_path)
            self.validate(test_imgs , test_labs ,batch_size , 0 , False )
            pred_dic[model_path] =  self.pred_all

        f = open('ensemble_report.txt' , 'w')
        # Run all combinations

        max_acc = 0
        max_list = []
        for k in range(2, len(pred_dic.keys()) + 1):
            k_max_acc = 0
            k_max_list = []
            print 'K : {}'.format(k)


            for cbn_models in itertools.combinations(pred_dic.keys(), k):
                for idx, model in enumerate(cbn_models):
                    pred = pred_dic[model]
                    pred = np.asarray(pred)
                    if idx == 0:
                        pred_sum = pred
                    else:
                        pred_sum += pred
                pred_sum=np.asarray(pred_sum)
                pred_sum = pred_sum / float(len(cbn_models))
                acc = self.get_acc(preds = pred_sum, trues=test_labs)
                if max_acc < acc:
                    max_acc = acc
                    max_pred = pred_sum
                    max_list = cbn_models
                if k_max_acc < acc:
                    k_max_acc = acc
                    k_max_list = cbn_models
        print 'Max Acc : ', max_acc
        print 'Max Combination model : ', max_list
        print f.write('Model List : {}\n'.format(max_list))
        print f.write('max Acc : {}\n'.format(max_acc))
        for pred in max_pred:
            print pred
            f.write(str(pred[0])+'\t'+str(pred[1])+'\n')
        return max_acc , max_pred , max_list


if __name__ =='__main__':
    """"""
    imgs = []
    for dirpath , subdir , files in os.walk('./my_data/abnormal'):
        for f in files:
            img = np.asarray(Image.open(os.path.join(dirpath , f)))
            imgs.append(img)
    tester = Tester(None)
    # Black Box Test
    """
    test_imgs = np.asarray(imgs)
    
    imgs=tester.black_box(img , 150)
    for i in imgs:
        coord , img =imgs[i]
        print np.shape(img)
    """
    # Mask Test
    #tester.mask(img , 0.5 , '<' ,)

    start_time =time.time()
    model_path = 'models/resnet_18/best_model/model-37620'
    test_imgs=np.load('my_data/cacs_abnormal_100_inf.npy')[:]

    test_imgs=test_imgs/255.
    test_labs=np.zeros([len(test_imgs) , 2])
    test_labs[:,0]=1
    batch_size = 60
    tester=Tester(None)
    pred_all, acc, loss=tester.eval(model_path, test_imgs, test_labs, batch_size , 'tmp_actmap' )
    test_labs=np.argmax(test_labs , axis=1)
    tester.plotROC(predStrength = pred_all[:,1] , labels = test_labs , prefix = 'CAC fundus Classifier', savepath='./tmp.png')

    print pred_all
    print start_time - time.time()
    print acc


    #Ensemble
    tester.ensemble(test_imgs, test_labs, 60, './models/best_models/0_from_5555', './models/best_models/0_from_5566',
                    './models/best_models/0_from_5571', './models/best_models/1_from_5555' , './models/best_models/1_from_5571')

"""
    

    #test_imgs=np.load('my_data/abnormal_test.npy')[:2]
    test_imgs=test_imgs/255.
    test_labs=np.zeros([len(test_imgs) , 2])
    test_labs[:,1]=1
    batch_size = 60
    tester=Tester(None)
    pred_all, acc, loss=tester.eval(model_path, test_imgs, test_labs, batch_size ,'tmp_actmap')
    print acc
    tf.reset_default_graph()
"""


