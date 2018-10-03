import skimage
import skimage.io as ski
import skimage.filters as skif
import skimage.transform as skit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np


## INPUT IMAGE
img1 = ski.imread("E:\Blood group\s.jpg")


##Color Plane Extraction: RGB Green Plane
_,img2,_ = cv2.split(img1)


##Auto Threshold: Clustering
t = skif.threshold_otsu(img2, nbins=256)
print(t)
#plt.imshow(g,cmap="gray")