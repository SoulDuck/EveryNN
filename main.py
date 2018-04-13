from VGG import VGG
from Recorder import Recorder
from Trainer import Trainer
from Tester import Tester
model_name = 'vgg_11'
vgg = VGG(model_name, True, logit_type='fc', n_classes=10)
recorder = Recorder(folder_name=model_name)
trainer = Trainer(recorder)
tester=Tester(recorder)




