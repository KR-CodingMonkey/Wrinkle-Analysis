import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def switch(x):
    return{
        '46-50' : 1,
        '51-55' : 2,
        '56-60' : 3,
        '61-65' : 4,
        '66-70' : 5,
        '71-75' : 6,
        '76-80' : 7
        }.get(x, "default")
def append(val_1, val_2, per, i):
    
    if i == 1:
        sobely_46.append(val_1)
        sobelx_46.append(val_2)
        per_46.append(per)
    elif i == 2:
        sobely_51.append(val_1)
        sobelx_51.append(val_2)
        per_51.append(per)
    elif i == 3:
        sobely_56.append(val_1)
        sobelx_56.append(val_2)
        per_56.append(per)
    elif i == 4:
        sobely_61.append(val_1)
        sobelx_61.append(val_2)
        per_61.append(per)
    elif i == 5:
        sobely_66.append(val_1)
        sobelx_66.append(val_2)
        per_66.append(per)
    elif i == 6:
        sobely_71.append(val_1)
        sobelx_71.append(val_2)
        per_71.append(per)
    elif i == 7:
        sobely_76.append(val_1)
        sobelx_76.append(val_2)
        per_76.append(per)
    else :
        return
def myPrint(sobelx, sobely, per, str):
    print("{0} :".format(str))

    print ("Xparam_{0} : {1}".format(str, np.mean(sobelx))) # average
    print ("Yparam_{0} : {1}".format( str, np.mean(sobely)))
    print ("Pparam_{0} : {1}".format(str, np.mean(per)))
    print("\n")

    print ("Xparam_{0} : {1}".format(str, np.std(sobelx))) #표준편차
    print ("Yparam_{0} : {1}".format(str, np.std(sobely)))
    print ("Pparam_{0} : {1}".format( str, np.std(per)))
    print("\n")

path = 'C:\\Users\\Kim\\Desktop\\PhotoByAge\\'
classes = ['46-50', '51-55', '56-60']
#classes = ['46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80']

sobely_46 = []
sobelx_46 = []
per_46 = []

sobely_51 = []
sobelx_51 = []
per_51 = []

sobely_56 = []
sobelx_56 = []
per_56 = []

sobely_61 = []
sobelx_61 = []
per_61 = []

sobely_66 = []
sobelx_66 = []
per_66 = []

sobely_71 = []
sobelx_71 = []
per_71 = []

sobely_76 = []
sobelx_76 = []
per_76 = []

sobely_kerut = []
sobelx_kerut = []
sobely_datar = []
sobelx_datar = []
per_kerut = []
per_datar = []

msg = "tipe: {0}, sobel_y: {1:1.6}, sobel_x: {2:1.6}, per: {3:1.6}"
    
bgdModel = np.zeros((1, 65), np.float64) # 배경에 속해있는 픽셀
fgdModel = np.zeros((1, 65), np.float64) # 전경에 속해있는 픽셀
        
for i in classes:
    file_path = os.path.join(path, i, '*.jpg')
    files = glob.glob(file_path)
    for j in files:
        image = cv2.imread(j)
        height, width, _= image.shape
        rect = (int(width * 0.15), int(height * 0.15), int(width * 0.7), int(height * 0.7)) # 전경을 포함하고 있는 사각영역(x,y,w,h)

        if(np.sum(image) == None): continue
        else :
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.GaussianBlur(image,(5,5),0)
        
            mask = np.zeros(image.shape[:2], np.uint8) # 전경일것같은 마스크 이미지
            cv2.grabCut(image, mask, rect, bgdModel,fgdModel, 10, cv2.GC_INIT_WITH_RECT) # iterCount - 반복횟수
        
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            image = image * mask2[:,:,np.newaxis] # np.new 차원증가
        
            w, h = image.shape[:2]
            count = 1
        
            for m in range (w):
                for n in range (h):
                    if(mask2[m][n] != 0):
                        count += 1
       
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            sobely = cv2.Sobel(image_gray, cv2.CV_8UC1, 0, 1, ksize=5)
            sobelx = cv2.Sobel(image_gray, cv2.CV_8UC1, 1, 0, ksize=5)

            #print (j)
            #plt.imshow(sobelx, cmap = 'gray')
            #plt.show()

        
            height, width= sobely.shape
            val_1 = cv2.sumElems(sobely)[0]/(count)
            val_2 = cv2.sumElems(sobelx)[0]/(count)
        
            age = switch(i)
            if(val_1 <= 1 or val_2 <= 1 or val_1 > 255 or val_2 > 255) : continue
            else :
                per = val_1/val_2
                append(val_1, val_2, per, age)
                print (msg.format(j, val_1, val_2, val_1/val_2))

    print("\n\n\n\n")
    myPrint(sobelx_46, sobely_46, per_46, '46-50')
    myPrint(sobelx_51, sobely_51, per_51, '51-55')
    myPrint(sobelx_56, sobely_56, per_56, '56-60')
    myPrint(sobelx_61, sobely_61, per_61, '61-65')
    myPrint(sobelx_66, sobely_66, per_66, '66-70')
    myPrint(sobelx_71, sobely_71, per_71, '71-76')
    myPrint(sobelx_76, sobely_76, per_76, '76-80')


print("\n\n\n\n")
myPrint(sobelx_46, sobely_46, per_46, '46-50')
myPrint(sobelx_51, sobely_51, per_51, '51-55')
myPrint(sobelx_56, sobely_56, per_56, '56-60')
myPrint(sobelx_61, sobely_61, per_61, '61-65')
myPrint(sobelx_66, sobely_66, per_66, '66-70')
myPrint(sobelx_71, sobely_71, per_71, '71-76')
myPrint(sobelx_76, sobely_76, per_76, '76-80')



