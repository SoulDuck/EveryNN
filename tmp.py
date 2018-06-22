#-*- coding:utf-8 -*-
import cv2 , glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
scale =300
import aug
"""
for f in glob.glob("train/*.jpeg")+ glob.glob("test/*.jpeg"):
    try:
        a = cv2.imread(f)
        #s c a l e img t o a gi v e n r a di u s
        a=scaleRadius(a , scale)
        #subtract local mean color
        a=cv2.addWeighted( a , 4 , cv2.GaussianBlur(a,(0 ,0) , scale/30)  −4 ,128)
        #remove o u t e r 10%
        b=numpy.zeros(a.shape)
        cv2.circle( b , ( a.shape[1] / 2 , a.shape[0] / 2 ) , int( scale * 0.9) , ( 1 , 1 , 1 ) , −1 , 8 , 0)
        a = a∗b + 128∗(1−b )
        cv2.imwrite(str( scale )+"_"+f , a )
"""



if '__main__' == __name__:
    img = np.asarray(Image.open('./tmp.png'))
    print np.shape(img)
    clahe_image =aug.clahe_equalized(img)

    merge_img=aug.fundus_projection(img , 300)

    clahe_image = aug.clahe_equalized(img)
    merge_img=merge_img/255.

    plt.imshow(clahe_image)
    plt.show()

    plt.imshow(img)
    plt.show()
    plt.imsave('merge_img_tmp.png' , merge_img)
    img=Image.open('merge_img_tmp.png')
    print np.max(img)
    print np.shape(img)

    plt.imshow(img)
    plt.show()


