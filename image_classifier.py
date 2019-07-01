# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 16:25:08 2019

@author: zaki
"""

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


train_dir='C:/Users/zaki/Videos/chest_xray/train'
test_dir='C:/Users/zaki/Videos/chest_xray/test'




train_pnemo = []
train_norm=[]
for path, subdirs, files in os.walk('C:/Users/zaki/Videos/chest_xray/train/'):
       for name in files:
           if 'pnemonia' in path:
               train_pnemo.append(os.path.join(path, name))
           elif 'normal' in path:
               train_norm.append(os.path.join(path,name))
                
train_imgs=train_pnemo+train_norm
random.shuffle(train_imgs)
import matplotlib.image as mpimg
for ima in train_imgs[0:3]:
    img=mpimg.imread(ima)
    imgplot=plt.imshow(img)
    plt.show()

thumb_size=((150,150))
channels=1

def read_image(list_of_images):
    X=[]
    y=[]
    
    for f in list_of_images:
            
            imag=Image.open(f,mode='r').convert('L')
            mimg=imag.resize(thumb_size,Image.ANTIALIAS)
            np_img=np.array(mimg)
#            print(np_img.shape)
            X.append(np_img)
            
            
            if 'pnemonia' in f:
                y.append(1)
            elif 'normal' in f:
                y.append(0)
        
    return X,y
#
X,y=read_image(train_imgs)
#
plt.figure(figsize=(20,10))
columns=5
for i in range(columns):
    plt.subplot(5/columns+1,columns,i+1)
    plt.imshow(X[i])
X=np.asarray(X)
y=np.asarray(y)

import seaborn as sns
#sns.countplot(y)
#plt.title("Labels for pnemo and norm")
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=20,random_state=2)
ntrain=len(X_train)
nval=len(y_train)
batch_size=32
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array,load_img

from keras.layers import Dense, Activation
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(150, 150)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train_datagen=ImageDataGenerator(rescale=1./255,
#                                 rotation_range=40,
#                                 width_shift_range=0.2,
#                                 height_shift_range=0.2,
#                                 shear_range=0.2,
#                                 zoom_range=0.2,
#                                 horizontal_flip=True,)
#val_datagen=ImageDataGenerator(rescale=1./255)
#
#train_generator=train_datagen.flow(X_train,y_train,batch_size=batch_size)
#val_generator=val_datagen.flow(X_val,y_val,batch_size=batch_size)
#
#model.fit_generator(train_generator,steps_per_epoch=ntrain//batch_size,
#                            epochs=32,validation_data=val_generator,
#                            validation_steps=nval//batch_size)

model.fit(X_train,y_train, epochs=100)
#score=model.evaluate(X_train,y_train)
pred = model.predict_classes(X_val)
from sklearn.metrics import classification_report, confusion_matrix
conf=confusion_matrix(y_val, pred)
test_loss, test_acc = model.evaluate(X_val,y_val)

print('Test accuracy:', test_acc)
