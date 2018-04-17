from VGG import VGG
from RESNET_V1 import RESNET_V1
from Recorder import Recorder
from Trainer import Trainer
from Tester import Tester
import tensorflow as tf



model_name = 'vgg_11'
vgg = VGG('sgd' , True , True,   model_name, 'fc'  , 'cifar10')


model_name = 'resnet_18'
resnet_v1 = RESNET_V1('sgd' , True , True , model_name , 'gap' , 'cifar_10')


recorder = Recorder(folder_name=model_name)
trainer = Trainer(recorder)
tester=Tester(recorder)
test_imgs=tester.pipeline.test_imgs
test_labs=tester.pipeline.test_labs
batch_size = 60
global_step=0

for i in range(10):
    mean_acc, mean_loss, pred_all=tester.validate(test_imgs, test_labs , batch_size ,global_step)
    print '\t Acc :{}  Loss :{} '.format(mean_acc , mean_loss)
    global_step = trainer.training(1000,global_step ,batch_size)

tf.reset_default_graph()
