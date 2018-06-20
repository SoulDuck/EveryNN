import tensorflow as f
import Tester
from Dataprovider import Dataprovider
import numpy as np
import os
from utils import cls2onehot
dirpath = '/home/mediwhale-5/PythonProjects/everyNN/my_data/project10'
test_data = os.path.join(dirpath ,'test_nor_0_10_abnor_10_inf.tfrecord')
val_data = os.path.join(dirpath ,'val_nor_0_10_abnor_10_inf.tfrecord')
#restore_model = '/Users/seongjungkim/PycharmProjects/everyNN/models/resnet/4/model-26730'
restore_model = '/home/mediwhale-5/PythonProjects/everyNN/models/resnet_18/4/model-26730'
tester=Tester.Tester(None)

tester._reconstruct_model(restore_model)
#tester.validate( save_model = None )
test_imgs  , test_labs , test_fs = Dataprovider.reconstruct_tfrecord_rawdata(test_data , None )
val_imgs ,val_labs , val_fs = Dataprovider.reconstruct_tfrecord_rawdata(val_data , None)
val_labs=cls2onehot(val_labs ,2)
test_labs=cls2onehot(test_labs ,2)
tester.n_classes =2
tester.validate(val_imgs ,val_labs , 60 , 0 , None )





