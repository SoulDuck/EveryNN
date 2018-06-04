import os
import numpy as np
from PIL import Image
import Dataprovider
path = '/Volumes/Seagate Backup Plus Drive/IMAC/project6/val.tfrecord'
path = '/Volumes/Seagate Backup Plus Drive/IMAC/project6/test.tfrecord'
imgs ,labs , fnames = Dataprovider.Dataprovider.reconstruct_tfrecord_rawdata(path , resize=None)

print labs
labs=np.asarray(labs)
indices=[labs==0]
print labs[indices]
indices=np.where([labs==1])[1]
print labs[indices]