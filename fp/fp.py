import os,sys
import cv2
import numpy as np
from PIL import *
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from keras.optimizers import RMSprop
fn_dir='/home/djohan/code/datamining/Data'

(images, lables, names, id) = ([],[],{},0)
for (subdirs,dirs,files) in os.walk(fn_dir):
    #print files
    for subdir in dirs:
        names[id]=subdir
        mypath=os.path.join(fn_dir,subdir)
        #print mypath
        for item in os.listdir(mypath):
            if '.png' in item:
                label=id
                image=cv2.imread(os.path.join(mypath,item),0)
                r_image=cv2.resize(image,(30,30)).flatten()
                if image is not None:
                    images.append(r_image)
                    lables.append(names[id])
        id=id+1
        
        
(images,lables)=[np.array(lis) for lis in [images,lables]]

nf=800
pca=PCA(n_components=nf)
img_feature=pca.fit_transform(images)

classifier=SVC(verbose=0,kernel='poly',degree=3)
classifier.fit(img_feature,lables)

test_image=cv2.imread('/home/djohan/code/datamining/A.png',0)
test_arr_image=cv2.resize(test_image,(30,30))
test_arr_image1=np.array(test_arr_image).flatten()
test_arr_image1=test_arr_image1.reshape(1,-1)

im_test=pca.transform(test_arr_image1)
pred=classifier.predict(im_test)
print (pred)
#print len(images)
#cv2.imshow('image',images[10])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#plt.imshow(images[10], cmap = 'gray', interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()

