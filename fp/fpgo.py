#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:05:22 2018

@author: djohan
"""

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import cv2
from sklearn.model_selection import train_test_split
fn_dir='/home/djohan/code/datamining/Data'
nama_kelas=['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
(images, lables, names, id) = ([],[],{},0)
for (subdirs,dirs,files) in os.walk(fn_dir):
    #print files
    for subdir in dirs:
        names[id]=subdir
        mypath=os.path.join(fn_dir,subdir)
        for item in os.listdir(mypath):
            if '.png' in item:
                label=id
                image=cv2.imread(os.path.join(mypath,item),0)
                r_image=cv2.resize(image,(50,50))
                r_image=r_image/255
                if image is not None:
                    images.append(r_image)
                    lables.append(names[id])
        id=id+1
(images,lables)=[np.array(lis) for lis in [images,lables]]
#r_image=cv2.resize(image,(30,30))
#r_image=r_image/255
fig = plt.figure()
plt.subplot(2,1,1)
plt.imshow(images[40], cmap='gray', interpolation='none')
plt.title("Class {}".format(lables[40]))
plt.xticks([])
plt.yticks([])
plt.subplot(2,1,2)
plt.hist(images[0].reshape(2500))
plt.title("Pixel Value Distribution")
fig
X_train, X_test, y_train, y_test = train_test_split(images, lables, test_size=0.2, random_state=42)

y_train = y_train.astype(np.float)
y_test = y_test.astype(np.float)

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

X_train = X_train.reshape(1584, 2500)
X_test = X_test.reshape(396, 2500)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print(np.unique(y_train, return_counts=True))

n_classes = 36
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

model = Sequential()
model.add(Dense(1700, input_shape=(2500,)))
model.add(Activation('relu'))                            
model.add(Dropout(0.2))

model.add(Dense(1700))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(36))
model.add(Activation('softmax'))

model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
history = model.fit(X_train, Y_train,
          batch_size=128, epochs=300,
          verbose=2,
          validation_data=(X_test, Y_test))

save_dir = "/home/djohan/code/datamining/results/"
model_name = 'FP.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

fig


mnist_model = load_model('/home/djohan/code/datamining/results/FP.h5')
c=mnist_model.get_weights()
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])

predicted_classes = mnist_model.predict_classes(X_test)
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

plt.rcParams['figure.figsize'] = (7,14)

figure_evaluation = plt.figure()

for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    b=y_test[correct]
    b=b.astype(np.int64)
    c=predicted_classes[correct]
    plt.imshow(X_test[correct].reshape(50,50), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(nama_kelas[c], nama_kelas[b]))
    plt.xticks([])
    plt.yticks([])
    
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    b=y_test[incorrect]
    b=b.astype(np.int64)
    c=predicted_classes[incorrect]
    plt.imshow(X_test[incorrect].reshape(50,50), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(nama_kelas[c], nama_kelas[b]))
    plt.xticks([])
    plt.yticks([])

figure_evaluation