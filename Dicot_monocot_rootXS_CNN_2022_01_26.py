#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN for classifying dicot vs monocot root cross sections
Author: Michael Gruenstaeudl, PhD | Email: m_gruenstaeudl@fhsu.edu
"""

import numpy as np
from glob import glob
from cv2 import imread
from tensorflow.keras import layers, models, preprocessing

img_w, img_h, batch, epochs = 150, 150, 10, 10
train_dir, test_dir = "data/train/", "data/test/"

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation="relu",input_shape=(img_w,img_h,3)),
    layers.MaxPooling2D((2,2)), layers.Dropout(0.25),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)), layers.Dropout(0.25),
    layers.Flatten(), layers.Dense(64,activation="relu"),
    layers.Dropout(0.5), layers.Dense(1,activation="sigmoid")
])
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])

datagen = preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split=0.2)
train_gen = datagen.flow_from_directory(train_dir,target_size=(img_w,img_h),batch_size=batch,class_mode="binary",subset="training")
val_gen = datagen.flow_from_directory(train_dir,target_size=(img_w,img_h),batch_size=batch,class_mode="binary",subset="validation")

model.fit(train_gen,epochs=epochs,validation_data=val_gen)

test_imgs=[imread(p)[0:img_w,0:img_h] for p in glob(test_dir+"*.jpg")]
test_imgs=np.array(test_imgs).astype("float32")/255.0
model.evaluate(test_imgs)
print("Predictions:",model.predict(test_imgs))
