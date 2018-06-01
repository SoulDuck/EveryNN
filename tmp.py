import os
import numpy as np
from PIL import Image
#root_dir = '/Volumes/Seagate Backup Plus Drive/data/project_000005/new_calc_fundus_350/1year/abnormal'
root_dir = '/Volumes/Seagate Backup Plus Drive/data/project_000005/1year/350/abnormal'
imgs = []
for dirpath , subdirs , files in os.walk(root_dir):
    for f in files:
        img_path = os.path.join(dirpath,  f)
        img = np.asarray(Image.open(img_path))
        imgs.append(img)

imgs=np.asarray(imgs)
np.save('/Users/seongjungkim/PycharmProjects/everyNN/my_data/cacs_abnormal_100_inf.npy' , imgs)



