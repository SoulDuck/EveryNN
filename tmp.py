import os
import numpy as np
from PIL import Image
root_dir = '/Volumes/Seagate Backup Plus Drive/data/project0000004/new_calc_fundus_350/1year/abnormal_100_inf'
imgs = []
for dirpath , subdirs , files in os.walk(root_dir):
    for f in files:
        img_path = os.path.join(dirpath,  f)
        img = np.asarray(Image.open(img_path))
        imgs.append(img)

imgs=np.asarray(imgs)
np.save('/Users/seongjungkim/PycharmProjects/everyNN/my_data/new_cacs_abnormal_100_inf.npy' , imgs)

