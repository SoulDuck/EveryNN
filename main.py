from VGG import VGG
from INCEPTION_V4 import INCEPTION_V4
from RESNET_V1 import RESNET_V1
from Recorder import Recorder
from Trainer import Trainer
from Tester import Tester
import tensorflow as tf
import cifar
import my_data
test_imgs , test_labs =my_data.get_test_imgs_labs()


model_name = 'vgg_11'
vgg = VGG('sgd' , False , True,   model_name, 'gap'  , 'mydata' , 60 , resize=(300,300) , num_epoch=100)
recorder = Recorder(folder_name=model_name)
trainer = Trainer(recorder)
tester=Tester(recorder)

global_step=0
for i in range(10):
    mean_acc, mean_loss, pred_all=tester.validate(test_imgs, test_labs , 60,global_step)
    print '\t Acc :{}  Loss :{} '.format(mean_acc , mean_loss)
    global_step = trainer.training(1000,global_step)
