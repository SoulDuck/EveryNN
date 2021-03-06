import Tester
import numpy as np
from PIL import Image
import aug
import utils
def random_rotate_90_180_270(images , k ):
    if k is None :
        k=np.random.randint(0,4)
    images=np.rot90(images , k , axes =(1,2))
    return images


def cls2onehot(cls , depth):

    labs=np.zeros([len(cls) , depth])
    for i,c in enumerate(cls):
        labs[i,c]=1
    return labs
restore_model  = './models/vgg_11/18/model-564'
tester=Tester.Tester(None)
tester._reconstruct_model(restore_model)

tester.n_classes =2
"""
best           second 
# [1,33](73)   [1,32](72)    1.png
# [13,43](875) [13,42](874) 3.png
# [3,20](146)  [3,21](147)   2.png

"""
imgs_list = []

for p in range(1,13):
    utils.show_progress(p, 12)
    for i in range(7):
        try:
            test_imgs = np.load('../Find_Wally/wally_raspCam_np/second/{}_{}.npy'.format(p,i))
            test_imgs = aug.apply_clahe(test_imgs)
            test_imgs = random_rotate_90_180_270(test_imgs , 3)

            #test_imgs = np.load('../Find_Wally/wally_raspCam/wally_1_1.npy')
            test_labs=[0]*len(test_imgs)
            test_labs=cls2onehot(test_labs ,2 )

            test_imgs = test_imgs/255.
            tester.validate(test_imgs , test_labs, 60 ,0 ,False)
            indices = np.where([np.asarray(tester.pred_all)[:,0] > 0.8])[1]
            print indices
            wally_imgs = test_imgs[indices]
            imgs_list.append(wally_imgs)
        except:
            print 'page : {} index : {}'.format(p,i)
            continue;

imgs=np.vstack(imgs_list)
np.save('wally_imgs.npy' , imgs)

