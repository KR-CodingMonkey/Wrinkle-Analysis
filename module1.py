import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread("C:\\Users\\Kim\\Desktop\\Scoring_Photo\\Old\\old_45.jpg")
mask = np.zeros(image.shape[:2], np.uint8) # 전경일것같은 마스크 이미지

bgdModel = np.zeros((1,65), np.float64) # 배경에 속해있는 픽셀
fgdModel = np.zeros((1, 65), np.float64) # 전경에 속해있는 픽셀

rect = (50, 50, 250, 250) # 전경을 포함하고 있는 사각영역(x,y,w,h)
cv2.grabCut(image, mask, rect, bgdModel,
           fgdModel, 5, cv2.GC_INIT_WITH_RECT) # iterCount - 반복횟수

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
image = image * mask2[:,:,np.newaxis] # np.new 차원증가

w, h = image.shape[:2]
count = 0

for i in range (w):
    for j in range (h):
        if(mask2[i][j] != 0):
            count += 1

print(count)
plt.imshow(image)
plt.colorbar()
plt.show()
