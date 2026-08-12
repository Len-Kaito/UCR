
import numpy as np 
# import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import os
from keras.utils import to_categorical
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split

#Lấy tất cả đường đẫn đến ảnh 
filenames=os.listdir("data_no_staight")
labels=[]
paths=[]
images=[]

for i in filenames:
    path=os.path.join("data_no_staight/{}".format(i))
    file_images=os.listdir(path)
    for j,f in enumerate(file_images):
        image=os.path.join(path,f)
        images.append(image)

#Xáo trộn đường dẫn
random.shuffle(images)


label = [p.split(os.path.sep)[-2] for p in images]
label = to_categorical(label, 6)
print(label)

X = []
for (j, imagePath) in enumerate(images):
    image = cv2.imread(imagePath)
    X.append(cv2.resize(image, (30,30)))
X = np.array(X)

X = X.astype('float32')/255 #đưa giá trị ảnh 0-255 về 0-1
X_train, X_valid, y_train, y_valid = train_test_split(X, label, test_size=0.2, random_state=0)  


earlystop = EarlyStopping(patience=15)
learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', 
                                            patience=4, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
checkpoint = ModelCheckpoint("models/model-{epoch:03d}.h5",
                                 monitor='accuracy',
                                 verbose=1,
                                 save_best_only=False,
                                 mode='auto')
callbacks = [earlystop,learning_rate_reduction,checkpoint]


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(30, 30, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(6, activation='softmax'))

model.compile(
    loss='categorical_crossentropy', 
    optimizer=Adam(0.001), 
    metrics=['accuracy']
)
model.summary()


epochs = 20
history = model.fit(X, label, batch_size=32, epochs=epochs, validation_data=(X_valid, y_valid),callbacks=callbacks)
model.save('last.h5')
