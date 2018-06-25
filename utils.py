import sys , math
import matplotlib.pyplot as plt
import random
import numpy as np

def cls2onehot(cls , depth):

    labs=np.zeros([len(cls) , depth])
    for i,c in enumerate(cls):
        labs[i,c]=1
    return labs
def show_progress(step, max_iter):
    msg = '\r progress {}/{}'.format(step, max_iter)
    sys.stdout.write(msg)
    sys.stdout.flush()


def plot_images(imgs , names=None , random_order=False , savepath=None , no_axis=True):
    h=math.ceil(math.sqrt(len(imgs)))
    fig=plt.figure()
    for i in range(len(imgs)):
        ax=fig.add_subplot(h,h,i+1)
        if random_order:
            ind=random.randint(0,len(imgs)-1)
        else:
            ind=i
        img=imgs[ind]
        plt.axis('off')
        plt.imshow(img)
        if not names==None:
            ax.set_xlabel(names[ind])
    if not savepath is None:
        plt.savefig(savepath)
    plt.show()
