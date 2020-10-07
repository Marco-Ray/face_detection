import cv2
import numpy as np
import tools


def detect(filename):
    # cv2级联分类器CascadeClassifier,xml文件为训练数据
    face_cascade = cv2.CascadeClassifier("E:/study/uestc/senior/DIP/program/face_detection/train/result/Haar_24x24.xml")
    # 读取图片
    img = cv2.imdecode(np.fromfile(filename,dtype=np.uint8),-1)
    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # 绘制人脸矩形框
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # 命名显示窗口
    cv2.namedWindow('people')
    # 显示图片
    cv2.imshow('people', img)
    # 设置显示时间,0表示一直显示
    cv2.waitKey(0)

    # # 保存图片
    # cv2.imwrite('cxks.png', img)

path = 'E:/study/uestc/senior/DIP/database/row_img/face_detection/30_Labeled Faces in the Wild Home/1'
img_list = tools.get_all_files(path,mode=1)
for img in img_list:
    detect(img)

# img = 'E:/study/uestc/senior/DIP/database/row_img/face_detection/30_Labeled Faces in the Wild Home/1/Aaron_Eckhart_0001.jpg'
# detect(img)


