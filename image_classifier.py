# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:37:47 2019

@author: zaki
"""

import cv2

from PIL import Image
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import gc
import matplotlib.pyplot as plt
#!pip install tensorflow
#oc=os.getcwd()
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


test_imgs='C:/Users/zaki/Downloads/animals/test/


train_bears = []
train_pandas=[]
for path, subdirs, files in os.walk('C:/Users/zaki/Downloads/animals/train'):
       for name in files:
           if 'bears' in path:
               train_bears.append(os.path.join(path, name))
           elif 'pandas' in path:
               train_pandas.append(os.path.join(path,name))

train_imgs=train_bears+train_pandas
random.shuffle(train_imgs)
               
nrows=150
ncolumns=150
channels=1


def read_img(list_of_images):

    X=[]
    y=[]

    for image in list_of_images:
        cf=(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(nrows,ncolumns),interpolation=cv2.INTER_CUBIC))
        X.append(cf)
        
        if 'bears' in image:
                y.append(1)
        elif 'pandas' in image:
                y.append(0)
                    
                return X,y           

X,y=read_img(train_imgs)

#X=np.array(X)
#y=np.array(y)
#
#plt.figure(figsize=(20,10))
#columns=5
#for i in range(columns):
#    plt.subplot(5/columns+1,columns,i+1)
#    plt.imshow(X[i])
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.20,random_state=2)

ntrain=len(X_train)
nval=len(y_train)
batch_size=32
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras import optimizers
#from keras import ImageDataGenerator
from keras import img_to_array,load_img
#
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

train_datagen=ImageDataGenerator(rescale=1./255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')
val_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow(X_train,y_train,batch_size=batch_size)
val_gen=val_datagen.flow(X_val,y_val,batch_size=batch_size)

MDL=model.fit_generator(train_generator,steps_per_epoch=ntrain//batch_size,
                        epochs=64,
                        validation_data=val_generator,
                        validation_steps=nval//batch_size)
model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')

X_test,y_test=read_img(test_imgs[0:10])
X=np.array(X_test)
pred=model.predict_classes(X_test)
from sklearn.metrics import classification_report, confusion_matrix
conf=confusion_matrix(y_test, pred)
