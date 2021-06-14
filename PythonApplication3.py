import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def Probability(value, mean, deviasi):   # 정규분포 함수
    n = math.exp(-0.5 * np.power(( value - mean) / deviasi, 2)) / math.sqrt(2 * 3.14 * np.power(deviasi,2))
    return n

def AgeGraph(data):
    y = data['X_mean']
    x = [46, 51, 56, 61, 66, 71, 80]
    y_std = data['X_std']

    plt.xlabel('Age')
    plt.ylabel('Wrinkle degree')

    plt.plot(x, y)
    plt.title('Wrinkle Analysis by Age')
    plt.show()
    print(y)

def Total_P(Yparam, Xparam, YX):

    data = np.array([
        (
        "46-50", 
        84.27906739488657, 89.81745405537724, 1.0849588425575372,
        13.601038353813319, 9.37070136468089, 0.15717665275002926),
        ("51-55",
         86.48108531111633, 91.01967768173274, 1.0697557316785362,
         15.344732043098302, 12.513004699328919, 0.1512709550240102),
         ("56-60",
          86.11758758459246, 91.04172011008923, 1.0764893935005837,
          14.29832754190376, 10.224768540002945, 0.15242011995252647),
          ("61-65",
           87.78986931006202, 90.83239423557029, 1.0477729733527952,
           12.901280331412565, 11.650117104667574, 0.14804883955828932),
           ("66-70",
            90.37271707382094, 90.34738193030078, 1.018187002604744,
            14.380972514201243, 10.568476829209482, 0.159574736359403),
            ("71-75",
             89.78030143396805, 90.767321125047, 1.026110013849184,
             19.163365919349552, 19.163365919349552, 0.14749466486977836),
             ("76-80",
             90.29344919631984, 90.56526671662688, 1.0185950116911133,
             14.108152527473862, 11.032108659912813, 0.14624419123034452)],
                   
                    dtype = [('Age', 'U10'), ('X_mean', 'f4'), ('Y_mean', 'f4'), ('YX_mean', 'f4'),
                                            ('X_std', 'f4'), ('Y_std', 'f4'), ('YX_std', 'f4')]) 
 
    result = 0
    age = ''
    
    for i in range (7):
        p1 = Probability(Yparam, data['Y_mean'][i], data['Y_std'][i])
        p2 = Probability(Xparam, data['X_mean'][i], data['X_std'][i])
        p3 = Probability(YX, data['YX_mean'][i], data['YX_std'][i])
        total = 0.5 * p1 * p2 * p3 * 1000
        #print(p1, p2, p3)
        print('{0} : {1}'.format(data['Age'][i], total))
        if result < total :
            age = data['Age'][i]
            result = total;
   
    print(result)
    print(age)


### Main def

bgdModel = np.zeros((1, 65), np.float64) # 배경에 속해있는 픽셀
fgdModel = np.zeros((1, 65), np.float64) # 전경에 속해있는 픽셀

image = cv2.imread('C:\\Users\\Kim\\Desktop\\photos\\10.png')
height, width, _= image.shape
rect = (int(width * 0.15), int(height * 0.15), int(width * 0.75), int(height * 0.7)) # 전경을 포함하고 있는 사각영역(x,y,w,h)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # opencv - BGR, matplotlib - RGB, 그래서 BGR을 RGB로 바꿔줘야함
image = cv2.GaussianBlur(image,(5,5),0) #size는 홀수

plt.imshow(image)
plt.show()
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
sobely = cv2.Sobel(image_gray, cv2.CV_8UC1, 0, 1, ksize=5) # 경계선 검출, x방향의 변화율은 세로줄이고 y방향의 변화율은 가로줄

height, width= sobely.shape
param_1 = cv2.sumElems(sobely)[0] / count # Calculates the sum of array elements.
print ("Y Sobel : ", param_1)

sobelx = cv2.Sobel(image_gray, cv2.CV_8UC1, 1, 0, ksize=5) # x축 경계선
param_2 = cv2.sumElems(sobelx)[0]/count
print ("X Sobel : ", param_2)

param_3 = param_1/param_2

Total_P(param_1,param_2,param_3)
