import Tester
import numpy as np
from PIL import Image
def cls2onehot(cls , depth):

    labs=np.zeros([len(cls) , depth])
    for i,c in enumerate(cls):
        labs[i,c]=1
    return labs
restore_model  = './models/vgg_simple/0/model-9999'
tester=Tester.Tester(None)
tester._reconstruct_model(restore_model)

tester.n_classes =2
"""
best           second 
# [1,33](73)   [1,32](72)    1.png
# [13,43](875) [13,42](874) 3.png
# [3,20](146)  [3,21](147)   2.png

"""

test_imgs = np.load('val_imgs.npy')
#test_imgs = np.load('../Find_Wally/wally_raspCam/wally_1_1.npy')
test_labs=[0]*len(test_imgs)
test_labs=cls2onehot(test_labs ,2 )


test_imgs = test_imgs
tester.validate(test_imgs , test_labs, 60 ,0 ,False)
indices = np.where([np.asarray(tester.pred_all)[:,0] > 0.5])[1]
print indices
wally_imgs = test_imgs[indices]
np.save('wally_imgs.npy' , wally_imgs)
