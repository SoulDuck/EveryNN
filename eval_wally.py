import Tester
import numpy as np

restore_model  = './models/vgg_13/0/model-990'
tester=Tester.Tester(None)
tester._reconstruct_model(restore_model)
tester.n_classes =2

"""
best           second 
# [1,33](73)   [1,32](72)    1.png
# [13,43](875) [13,42](874) 3.png
# [3,20](146)  [3,21](147)   2.png

"""

test_imgs = np.load('./wally_data/1.npy')
test_labs = np.zeros([len(test_imgs)])
test_labs[73,72]=1

print test_labs

test_imgs = test_imgs/255.
tester.validate(test_imgs , test_labs, 60 ,0 ,False)

