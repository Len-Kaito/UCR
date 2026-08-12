from glob import glob
import cv2
import os
import shutil
import numpy as np
from random import shuffle

imgs = glob("img/*")[:500]
shuffle(imgs)


for k, img in enumerate(imgs):
    if 'flip_blur' in img or 'blur.png' in img:
        continue

    im =cv2.imread(img)
    h, w = im.shape[:2]

    ymin = np.random.randint(50, h-90)
    xmin = np.random.randint(50, w-90)
    crop = im[ymin:ymin+90, xmin: xmin+90]
    h, w = crop.shape[:2]
    if h > w:
        new_data = np.zeros((h, h, 3), np.uint8)
        c = ((h-w)//2)
        new_data[:, c:c+w] = crop.copy()

    else:
        new_data = np.zeros((w, w, 3), np.uint8)
        c = ((w-h)//2)
        new_data[c:c+h, :] = crop.copy()
    new_data = cv2.resize(
        new_data, (30, 30))


    cv2.imwrite(f"data_final/6/{k}.png", crop)


print(len(os.listdir('data_final/6/')))