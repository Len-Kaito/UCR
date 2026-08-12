from glob import glob
import cv2
import os
import shutil
import numpy as np
labels = sorted(glob("label/*.txt"))
imgs = sorted(glob("img/*"))


def unconvert(class_id, width, height, x, y, w, h):

    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)

shutil.rmtree("data_final/")
os.mkdir("data_final/")
for i in range(6):
    os.mkdir(f"data_final/{i}")
for k, (label, img) in enumerate(zip(labels, imgs)):
    if 'flip_blur' in img or 'blur.png' in img:
        continue

    id = open(label, 'r').readlines()[0].split()[0]
    cords = open(label, 'r').readlines()[0].split()[1:]
    im =cv2.imread(img)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    H, W = im.shape[0], im.shape[1]


    class_id, xmin, xmax, ymin, ymax = unconvert(id, W, H, float(cords[0]), float(cords[1]), float(cords[2]), float(cords[3]))
    crop = im[ymin:ymax, xmin: xmax]

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
    print(new_data.shape)
    if new_data.shape[:2] != (30, 30):
        print('-------')

    cv2.imwrite(f"data_final/{class_id}/{img.split('/')[-1][:-4]}.png", new_data)
    # print(id, label, im.shape, cords)

# img = cv2.imread("img/camtrai_0_flip.jpg")
# cv2.imwrite("img.png", img)
# print(unconvert(0, 416, 416, 0.27578125, 0.5326388888888889, 0.09062500000000001, 0.12916666666666668))

# crop = img[194:248, 95: 133]
# cv2.imwrite("crop.png", crop)