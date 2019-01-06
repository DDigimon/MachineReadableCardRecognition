from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import sys, urllib, json
import urllib.request
import urllib.parse
import base64
url = 'http://apis.baidu.com/idl_baidu/ocridcard/ocridcard'
import pytesseract
from PIL import Image

#手动输入值确定卷子类型,默认只有选择题
s=1
data=[]
'''
第一类卷子：
上边缘：350~450
下边缘：1411~1450
机读卡填土面积：50*10
x:[(90,160,270,350.410),(510,560,680,760,840),(930,1025,1120,1200,1280),(1360,1460,1550,1630,1730)]
y:[(500,520,560,610,650,680),(720,760,800,850,880,930),(950,1000,1030,1080,1130,1170),(1190,1220,1280,1325,1365,1405)]
'''
x0=[90,160,270,350.410,510,560,680,760,840,930,1025,1120,1200,1280,1360,1460,1550,1630,1730]
y0=[500,520,560,610,650,680,720,760,800,850,880,930,950,1000,1030,1080,1130,1170,1190,1220,1280,1325,1365,1405]
x1=[920,1000,1750,1150,1210,1275]
y1=[150,225,355,500,600]

xt1=[0,90,220,350,470,610,700,830,940,1070,1200,1300,1430,1540,1660,1790,1890,2015,2240,2270,2400]
yt1=[900,1000,1070,1140,1210,1300,1375,1450,1500,1580,1650,1750,1810,1880,1950,2150]

testnumx=[0,350,465,575,690,800,920]
IDnumx=[0,345,460,575,690,800,913,1025,1135,1243,1356,1469,1582,1691,1804,1920,2033,2146,2400]
answernumx=[0,244,2350,717,941,1190,1410,1668,1900,2152,2400]
y1num=[0,480,770]
y2num=[2250,2440,2700]

Answer=[]
#两种卷子统一规格
width0=1750
height0=1450
width1=2400
height1=2800



#图像膨胀
def Change(image,flag = 0,num = 2):

    w = image.width
    h = image.height
    size = (w,h)
    iChange = cv2.CreateImage(size,8,1)
    for i in range(h):
        for j in range(w):
            a = []
            for k in range(2*num+1):
                for l in range(2*num+1):
                    if -1<(i-num+k)<h and -1<(j-num+l)<w:
                        a.append(image[i-num+k,j-num+l])
            if flag == 0:
                k = max(a)
            else:
                k = min(a)
            iChange[i,j] = k
    return iChange
#灰度线性变化
def liner(test,pos):
    for x in range(test.shape[0]):
        for y in range(test.shape[1]):
            if test[x,y]>pos:
                test[x,y]=255
            else:
                test[x,y]=0
    return test
#卷子model0判题
def judgey0(y):
    if (y / 5 < 1):
        return  y + 1
    elif y / 5 < 2 and y/5>=1:
        return y % 5 + 20 + 1
    else:
        return y % 5 + 40 + 1
def judgex0(x):
    if(x%5==1):
        return 'A'
    elif(x%5==2):
        return 'B'
    elif(x%5==3):
        return 'C'
    elif(x%5==4):
        return 'D'
def judge0(x,y):
    if x/5<1 :
        #print(judgey0(y))
        return (judgey0(y),judgex0(x))
    elif x/5<2 and x/5>=1:
        #print(judgey0(y)+5)
        return (judgey0(y)+5,judgex0(x))
    elif x/5<3 and x/5>=2:
       # print(judgey0(y)+10)
        return (judgey0(y)+10,judgex0(x))
    else:
        #print(judgey0(y)+15)
        return (judgey0(y)+15,judgex0(x))

getid=[]
def get(data):
    testid = '000046'
    IDid = '51089209910423645X'
    answeid = '132204090507061080'
    getid.append(testid)
    getid.append(IDid)
    getid.append(answeid)

#数字识别
"""
template0=cv2.imread('E:\PyProgramma\pyImg\SummerTrain\source\img0\\1.jpg',0)
template1=cv2.imread('E:\PyProgramma\pyImg\SummerTrain\source\img1\\0.jpg',0)
template2=cv2.imread('E:\PyProgramma\pyImg\SummerTrain\source\img2\\0.jpg',0)
template3=cv2.imread('E:\PyProgramma\pyImg\SummerTrain\source\img3\\0.jpg',0)
template4=cv2.imread('E:\PyProgramma\pyImg\SummerTrain\source\img4\\0.jpg',0)
template5=cv2.imread('E:\PyProgramma\pyImg\SummerTrain\source\img5\\0.jpg',0)
template6=cv2.imread('E:\PyProgramma\pyImg\SummerTrain\source\img6\\0.jpg',0)
template7=cv2.imread('E:\PyProgramma\pyImg\SummerTrain\source\img7\\0.jpg',0)
template8=cv2.imread('E:\PyProgramma\pyImg\SummerTrain\source\img8\\0.jpg',0)
template9=cv2.imread('E:\PyProgramma\pyImg\SummerTrain\source\img9\\0.jpg',0)
#templatex=cv2.imread('E:\PyProgramma\pyImg\SummerTrain\source\imgx\\0.jpg',0)

'''
模板方法选择：
w,h = template.shape[::-1]  
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',  
      'cv2.TM_CCORR_NORMED', '*cv2.TM_SQDIFF', '*cv2.TM_SQDIFF_NORMED'] 
res = cv2.matchTemplate(img,template,method) 
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:  
  top_left = min_loc  
else:  
  top_left = max_loc  
  bottom_right = (top_left[0] + w, top_left[1] + h)  
  cv2.rectangle(img,top_left, bottom_right, 255, 2) 
'''

def templatematch(img,template):
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 左上角顶点
    topleft = min_loc
    # 右下角顶点
    bottom_right = (topleft[0] + w, topleft[1] + h)
    # 在图里绘制矩形
    #cv2.rectangle(img, topleft, bottom_right, (0, 0, 255), -1)
    #cv2.imshow('t',img)
    #cv2.waitKey(0)
    #print(topleft,bottom_right,min_val, max_val)
    return max_val

def Maxnum(img):
    for i in range(len(testnumx)-1):
        pro=[]
        tempimg=img[y1num[0]:y1num[1],testnumx[i]:testnumx[i+1]]
        #print(testnumx[i+1]-testnumx[i])
        pro.append(templatematch(tempimg,template0))
        pro.append(templatematch(tempimg, template1))
        pro.append(templatematch(tempimg, template2))
        pro.append(templatematch(tempimg, template3))
        pro.append(templatematch(tempimg, template4))
        pro.append(templatematch(tempimg, template5))
        pro.append(templatematch(tempimg, template6))
        pro.append(templatematch(tempimg, template7))
        pro.append(templatematch(tempimg, template8))
        pro.append(templatematch(tempimg, template9))
        #pro.append(templatematch(tempimg, templatex))
        print(pro.index(max(pro)))

        '''
        max=templatematch(tempimg,template0)
        if(max>templatematch(tempimg,template1)):
            print('0')
        else:
            print('1')
        '''


#Maxnum(NumImg)
'''
#单模板
w,h=template0.shape[::-1]
res=cv2.matchTemplate(NumImg,template0,cv2.TM_CCORR_NORMED)
min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
#左上角顶点
topleft=min_loc
#右下角顶点
bottom_right=(topleft[0]+w,topleft[1]+h)
#在图里绘制矩形
cv2.rectangle(NumImg,topleft,bottom_right,(0,0,255),-1)
print(min_val,max_val)
#def choose(Img):
'''
'''
testimage=Image.open('E:\PyProgramma\pyImg\SummerTrain\source\\test.png')
code=pytesseract.image_to_string(testimage)
print(code)
'''
"""
def congnitive(name):
    data = {}
    data['fromdevice'] = "pc"
    data['clientip'] = "10.10.10.0"
    data['detecttype'] = "LocateRecognize"
    data['languagetype'] = "ENG"
    data['imagetype'] = "1"
    # 图片在本地
    file_object = open('E:\PyProgramma\pyImg\SummerTrain\source\storetmp\\'+ name+'.png', 'rb')
    try:
        img = file_object.read()
    finally:
        file_object.close()
    data['image'] = base64.b64encode(img)

    decoded_data = urllib.parse.urlencode(data)
    decoded_data = decoded_data.encode('utf-8')
    # #print(decoded_data)
    req = urllib.request.Request(url, decoded_data)

    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("apikey", "fc5444de826f92f064d430d1ea095983")  # a63fd60067141f18cffa9af6a1563b4e

    resp = urllib.request.urlopen(req)  # , data = decoded_data)
    content = resp.read()
    if (content):
        content = json.loads(content.decode())
        with open('E:\PyProgramma\pyImg\SummerTrain\source\storetmp\\'+ name + '.json', 'w') as json_file:
            json_file.write(json.dumps(content))
            get(json_file)
        print(content)
def cutpic(img):
    tempimg1=img[240:461,213:939]
    tempimg1 = cv2.resize(tempimg1, (width, height), cv2.INTER_LANCZOS4)
    cv2.imwrite("E:\PyProgramma\pyImg\SummerTrain\source\storetmp\\T.png", tempimg1)
    tempimg2 = img[476:725, 221:1258]
    cv2.imwrite("E:\PyProgramma\pyImg\SummerTrain\source\storetmp\\TTTT.png", tempimg2)
    tempimg3 = img[476:725, 1258:2293]
    cv2.imwrite("E:\PyProgramma\pyImg\SummerTrain\source\storetmp\\TTTT2.png", tempimg3)
    tempimg4 = img[2236:2691, 100:379]
    cv2.imwrite("E:\PyProgramma\pyImg\SummerTrain\source\storetmp\\T1.png", tempimg4)
    tempimg5 = img[2236:2691, 597:864]
    cv2.imwrite("E:\PyProgramma\pyImg\SummerTrain\source\storetmp\\T2.png", tempimg5)
    tempimg6 = img[2236:2691, 1063:1337]
    cv2.imwrite("E:\PyProgramma\pyImg\SummerTrain\source\storetmp\\T3.png", tempimg6)
    tempimg7 = img[2236:2691, 1533:1808]
    cv2.imwrite("E:\PyProgramma\pyImg\SummerTrain\source\storetmp\\T4.png", tempimg7)
    tempimg8 = img[2236:2691, 2003:2281]
    cv2.imwrite("E:\PyProgramma\pyImg\SummerTrain\source\storetmp\\T5.png", tempimg8)

def read(name):
    with open('E:\PyProgramma\pyImg\SummerTrain\source\storetmp\\'+name+'.json',encoding='utf-8') as json_file:
        data = json.load(json_file)
        return data

# 加载图片，将它转换为灰阶，轻度模糊，然后边缘检测。
image = cv2.imread("E:\PyProgramma\pyImg\SummerTrain\source\\test10.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
bin=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,53,2)
blurred=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,2)
blurred=cv2.copyMakeBorder(blurred,5,5,5,5,cv2.BORDER_CONSTANT,value=(255,255,255))
edged = cv2.Canny(blurred, 10, 100)



'''
#hough transform
lines = cv2.HoughLinesP(edged,1,np.pi/180,30,minLineLength=5,maxLineGap=20)
lines1 = lines[:,0,:]#提取为二维
for x1,y1,x2,y2 in lines1[:]:
    cv2.line(image,(x1,y1),(x2,y2),(255,0,0),1)

thresh = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
plt.subplot(122),plt.imshow(thresh)
plt.xticks([]),plt.yticks([])
plt.show()

'''
'''
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image,contours,-1,(0,0,255),3)
cv2.imshow("img",image)
cv2.waitKey(0)
'''

# 从边缘图中寻找轮廓，然后初始化答题卡对应的轮廓
'''
findContours
image -- 要查找轮廓的原图像
mode -- 轮廓的检索模式，它有四种模式：
     cv2.RETR_EXTERNAL  表示只检测外轮廓                                  
     cv2.RETR_LIST 检测的轮廓不建立等级关系
     cv2.RETR_CCOMP 建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，
              这个物体的边界也在顶层。
     cv2.RETR_TREE 建立一个等级树结构的轮廓。
method --  轮廓的近似办法：
     cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max （abs (x1 - x2), abs(y2 - y1) == 1
     cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需
                       4个点来保存轮廓信息
      cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
'''
cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = None
# 确保至少有一个轮廓被找到
if len(cnts) > 0:
    # 将轮廓按大小降序排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # 对排序后的轮廓循环处理
    for c in cnts:
        # 获取近似的轮廓
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # 如果近似轮廓有四个顶点，那么就认为找到了答题卡
        if len(approx) == 4:
            docCnt = approx
            break
# 对原始图像和灰度图都进行四点透视变换
#cv2.drawContours(image, c, -1, (0, 0, 255), 5, lineType=0)
newimage=image.copy()
for i in docCnt:
    cv2.circle(newimage, (i[0][0],i[0][1]), 50, (255, 0, 0), -1)

plt.figure()
plt.subplot(131)
plt.imshow(image,cmap='gray')
#加上这句就隐藏坐标了
#plt.xticks([]), plt.yticks([])
plt.subplot(132)
plt.imshow(blurred,cmap = 'gray')
#plt.grid()
#plt.subplot(122),plt.plot([(0),(1000)])
#plt.xticks([]), plt.yticks([])
plt.subplot(133)
plt.imshow(newimage,cmap = 'gray')
#plt.grid()
#plt.subplot(122),plt.plot([(0),(1000)])
#plt.xticks([]), plt.yticks([])
plt.show()

paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))
# 对灰度图应用二值化算法
thresh=cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,53,2)
threshtmmp=thresh
#图形转换为标准方块
'''
插值方法
INTER_NEAREST - 最邻近插值
INTER_LINEAR - 双线性插值，这是默认的方法）
INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
INTER_CUBIC - 4x4像素邻域的双立方插值
INTER_LANCZOS4 - 8x8像素邻域的Lanczos插值
'''
#卷子类型为0
if s==0:
    thresh=cv2.resize(thresh,(width0,height0),cv2.INTER_LANCZOS4)
    paper=cv2.resize(paper,(width0,height0),cv2.INTER_LANCZOS4)
    #paper 用来标记边缘检测，所以建一个来保存
    paperorign=paper
    warped=cv2.resize(warped,(width0,height0),cv2.INTER_LANCZOS4)
    ChQImg=cv2.blur(thresh,(21,21))
    '''
    参数说明
    第一个参数 src    指原图像，原图像应该是灰度图。
   第二个参数 x      指用来对像素值进行分类的阈值。
   第三个参数 y      指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
   第四个参数 Methods  指，不同的不同的阈值方法，这些方法包括：
                •cv2.THRESH_BINARY        
                •cv2.THRESH_BINARY_INV    
                •cv2.THRESH_TRUNC        
                •cv2.THRESH_TOZERO        
                •cv2.THRESH_TOZERO_INV    
    '''
    ChQImg=cv2.threshold(ChQImg,120,225,cv2.THRESH_BINARY)
    ChQImg=ChQImg[1]

    #实验发现用边缘检测根本不靠谱……
    #确定选择题答题区
    #ChQImg=thresh[450:height0,0:width0]
    # 在二值图像中查找轮廓，然后初始化题目对应的轮廓列表
    cnts = cv2.findContours(ChQImg, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    '''
    检测用边缘绘制
    cv2.drawContours(paper, cnts[1], -1, (0, 0, 255), 10, lineType=0)
    '''
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    questionCnts = []
    # 对每一个轮廓进行循环处理
    for c in cnts:
        #计算轮廓的边界框，然后利用边界框数据计算宽高比
        (x,y,w,h)=cv2.boundingRect(c)
        questionCnts.append(c)
        if(w>50&h>10):
            questionCnts.append(c)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(paper, c, -1, (0, 0, 255), 5, lineType=0)
            cv2.circle(paper, (cX, cY), 7, (255, 255, 255), -1)
            Answer.append((cX,cY))

#卷子类型为1
elif s==1:
    thresh = cv2.resize(thresh, (width1, height1), cv2.INTER_LANCZOS4)
    paper = cv2.resize(paper, (width1, height1), cv2.INTER_LANCZOS4)
    # paper 用来标记边缘检测，所以建一个来保存
    paperorign = paper
    warped = cv2.resize(warped, (width1, height1), cv2.INTER_LANCZOS4)
    ChQImg = cv2.blur(thresh, (23, 23))
    '''
    参数说明
    第一个参数 src    指原图像，原图像应该是灰度图。
   第二个参数 x      指用来对像素值进行分类的阈值。
   第三个参数 y      指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
   第四个参数 Methods  指，不同的不同的阈值方法，这些方法包括：
                •cv2.THRESH_BINARY        
                •cv2.THRESH_BINARY_INV    
                •cv2.THRESH_TRUNC        
                •cv2.THRESH_TOZERO        
                •cv2.THRESH_TOZERO_INV    
    '''
    ChQImg = cv2.threshold(ChQImg, 100, 225, cv2.THRESH_BINARY)[1]

    NumImg=cv2.blur(thresh,(15,15))
    NumImg=cv2.threshold(NumImg, 170, 255, cv2.THRESH_BINARY)[1]
    #cv2.imwrite("E:\PyProgramma\pyImg\SummerTrain\source\\temp.jpg",NumImg)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    NumImg2 = cv2.dilate(warped, kernel)
    # NumImg2=cv2.adaptiveThreshold(ChQImg2,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
    NumImg2 = cv2.erode(NumImg2, kernel2)
    # NumImg2 = cv2.threshold(ChQImg2, 200, 225, cv2.THRESH_BINARY)[1]
    NumImg2 = cv2.adaptiveThreshold(NumImg2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 2)
    # ChQImg2 = cv2.threshold(ChQImg2, 200, 225, cv2.THRESH_BINARY)[1]
    NumImg2=NumImg2-NumImg
    NumImg2=cv2.blur(NumImg2,(5,5))
    NumImg2=cv2.threshold(NumImg2,170,255,cv2.THRESH_BINARY)[1]

    # 实验发现用边缘检测根本不靠谱……
    # 确定选择题答题区
    # ChQImg=thresh[450:height0,0:width0]
    # 在二值图像中查找轮廓，然后初始化题目对应的轮廓列表
    cnts = cv2.findContours(ChQImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    '''
    检测用边缘绘制
    cv2.drawContours(paper, cnts[1], -1, (0, 0, 255), 10, lineType=0)
    '''
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    questionCnts = []
    # 对每一个轮廓进行循环处理
    for c in cnts:
        # 计算轮廓的边界框，然后利用边界框数据计算宽高比
        (x, y, w, h) = cv2.boundingRect(c)
        questionCnts.append(c)
        if (w > 60 & h > 20)and y>900 and y<2000:
            questionCnts.append(c)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(paper, c, -1, (0, 0, 255), 5, lineType=0)
            cv2.circle(paper, (cX, cY), 7, (255, 255, 255), -1)
            Answer.append((cX, cY))

#答案选择输出
IDAnswer=[]
for i in Answer:
    for j in range(0,len(xt1)-1):
        if i[0]>xt1[j] and i[0]<xt1[j+1]:
            for k in range(0,len(yt1)-1):
                if i[1]>yt1[k] and i[1]<yt1[k+1]:
                    judge0(j,k)
                    IDAnswer.append(judge0(j,k))

    #print(i)
'''
newx1=[]
newy1=[]
for i in xt1:
    a=i%10
    if(a>5):
        i=(int(i/10))*10+10
    else:
        i=(int(i/10))*10
    newx1.append(i)
for i in yt1:
    a = i % 10
    if (a > 5):
        i = (int(i / 10)) * 10 + 10
    else:
        i = (int(i / 10)) * 10
    newy1.append(i)
newx1=list(set(newx1))
newy1=list(set(newy1))
newx1.sort()
newy1.sort()
print(Answer)
print(newx1)
print(len(newx1))
print(newy1)
print(len(newy1))
t.sort()
print(t)
'''
IDAnswer.sort()
print(IDAnswer)
print(len(IDAnswer))


#cutpic(NumImg)

#data=read('TTTT2')
get(data)
print(getid)
with open("E:\PyProgramma\pyImg\SummerTrain\source\\test.txt", "w") as f:
    f.write(str(IDAnswer)+'\n')
    f.write(str(getid)+'\n')

plt.figure()
plt.subplot(131)
plt.imshow(warped,cmap='gray')
#加上这句就隐藏坐标了
#plt.xticks([]), plt.yticks([])
plt.subplot(132)
plt.imshow(ChQImg,cmap = 'gray')
plt.subplot(133)
plt.imshow(NumImg,cmap = 'gray')
#plt.grid()
#plt.subplot(122),plt.plot([(0),(1000)])
#plt.xticks([]), plt.yticks([])
plt.show()

plt.figure()
plt.imshow(paper,cmap='gray')
#加上这句就隐藏坐标了
#plt.xticks([]), plt.yticks([])

#plt.grid()
#plt.subplot(122),plt.plot([(0),(1000)])
#plt.xticks([]), plt.yticks([])
plt.show()

