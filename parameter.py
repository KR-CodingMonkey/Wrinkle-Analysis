import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'C:\\Users\\Kim\\Desktop\\Analysis-of-facial-wrinkles-master\\Analysis-of-facial-wrinkles-master\\'
classes = ['berkerut', 'unwrinkled']

sobely_kerut = []
sobelx_kerut = []
sobely_datar = []
sobelx_datar = []
per_kerut = []
per_datar = []

msg = "tipe: {0}, sobel_y: {1:1.6}, sobel_x: {2:1.6}, per: {3:1.6}"

for i in classes:
    file_path = os.path.join(path, i, '*.jpg')
    files = glob.glob(file_path)
    for j in files:
        image = cv2.imread(j)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.GaussianBlur(image,(5,5),0)
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_RGB2GRAY)
        sobely = cv2.Sobel(image_gray, cv2.CV_8UC1, 0, 1, ksize=5)
        sobelx = cv2.Sobel(image_gray, cv2.CV_8UC1, 1, 0, ksize=5)
        
        height, width= sobely.shape
        val_1 = cv2.sumElems(sobely)[0]/(height * width)
        val_2 = cv2.sumElems(sobelx)[0]/(height * width)
        per = val_1/val_2
        
        if i == 'berkerut':
            sobely_kerut.append(val_1)
            sobelx_kerut.append(val_2)
            per_kerut.append(per)
        else:
            sobely_datar.append(val_1)
            sobelx_datar.append(val_2)
            per_datar.append(per)
        
        print (msg.format(j, val_1, val_2, val_1/val_2))

print (np.mean(sobely_kerut)) # average
print (np.std(sobely_kerut)) #표준편차

print (np.mean(sobelx_kerut))
print (np.std(sobelx_kerut))

print (np.mean(per_kerut))
print (np.std(per_kerut))

print (np.mean(sobely_datar))
print (np.std(sobely_datar))

print (np.mean(sobelx_datar))
print (np.std(sobelx_datar))

print (np.mean(per_datar))
print (np.std(per_datar))