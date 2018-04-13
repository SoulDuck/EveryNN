from VGG import VGG
from Recorder import Recorder
from Trainer import Trainer
from Tester import Tester
model_name = 'vgg_11'
vgg = VGG(model_name, True, logit_type='fc' , datatype='cifar10')
recorder = Recorder(folder_name=model_name)
trainer = Trainer(recorder)
tester=Tester(recorder)
test_imgs=tester.pipeline.test_imgs
test_labs=tester.pipeline.test_labs
batch_size = 60
global_step=0
global_step = trainer.training(10,global_step ,batch_size)
tester.validate(test_imgs, test_labs , batch_size ,global_step)




