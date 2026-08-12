import os
# from keras.models import load_model
import tensorflow as tf 

import cv2

import numpy as np 
# import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import os
from keras.utils import to_categorical
import random


model2 = tf.keras.models.load_model('models/model-016.h5')


X_test = cv2.imread('crop/5/6.png')
X_test=cv2.resize(X_test,(30, 30))
X_test = X_test.astype('float32')/255 

X_test = X_test.reshape(1,30,30,3)
predictions = model2.predict(X_test)
print(np.argmax(predictions[0]))