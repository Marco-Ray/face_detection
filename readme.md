# 使用opencv训练自己的人脸检测分类器
##目录
1. 安装opencv
2. 准备正负样本
3. 生成正负样本描述文件
4. 创建vec文件
5. 开始训练
6. 训练结束

## 1. 安装opencv
opencv虽然有java、python的集成版本，但貌似无法训练自己的分类器，因此去[opencv官网](https://opencv.org/releases/)下载c++编译的win10，下载完毕后点击exe文件安装即可。（注：4.0以上版本貌似移除了opencv_createsamples.exe和opencv_traincascade.exe，因此我下载了3.4.10版本）。

## 2. 准备正负样本
新建一个文件夹命名为train(可任意)，用以存放后续文件。
### 正样本：
正样本要求统一大小，且尽可能裁剪掉背景只保留人脸部分。我准备了4059个正样本，为了缩短训练时长，将其resize为24x24大小，并转为灰度图像，保存在train/pos_img文件夹下。在tools.py文件内有我编写好的resize、GrayImg、rename等函数。
### 负样本：
负样本不需要统一大小，只需要比正样本大，且不包含人脸即可。我准备了8432个负样本，且同样为了缩短训练时长，将其resize为50x50，并转换为灰度图像，保存在train/neg_img文件夹下。

## 3. 生成正负样本描述文件
### 生成负样本描述文件neg_img.txt
打开cmd窗口，输入一下命令进入neg_img文件夹内
```
>cd neg_img文件夹的绝对路径
```
我的情况里是
```
>E:
>cd E:\XXXX\face_detection\train\neg_img
```
先输入E:是因为cmd默认在C:而我的文件夹储存在E盘，需要先切换。

之后再输入以下命令
```
>dir /b >neg_img.txt
```
会在neg_img文件夹下生成一个neg_img.txt文件，内容如下
![neg_img_txt](E:/study/uestc/senior/DIP/database/ppt/neg_img_txt.png)
删除最后两行（空行和“neg_img.txt”），再用任意方式对每一行进行处理，最终结果需如下所示
![neg_img_txt_f](E:/study/uestc/senior/DIP/database/ppt/neg_img_txt_f.png)
将neg_img.txt存放至train文件下，与neg_img文件夹并列。

### 生成正样本描述文件pos_img.txt
前序步骤与生成neg_img.txt一致，只需将neg换为pos。但最终pos_img.txt内每一行的格式需如下所示
![pos_img_txt](E:/study/uestc/senior/DIP/database/ppt/pos_img_txt_f.png)
其中1表示图片内有一个人脸（两个则相应改为2），0 0 24 24表示人脸在图片中的四个坐标，若resize中的尺寸不为24x24则需相应修改。
同样最后把pos_img.txt存放在train文件夹下，与pos_img文件夹并列。

__至此训练数据准备完毕__

## 4. 创建vec文件
从opencv的安装目录下找到opencv_createsamples.exe和opencv_traincascade.exe文件，复制到train文件夹下。然后新建一个txt文件，内容如下
```
opencv_createsamples.exe -info pos_img.txt -vec pos.vec -num 4059 -w 24 -h 24
pause
```
并重命名为create_pos_samples.bat文件。
其中的-vec是指定后面输出vec文件的文件名，-info指定正样本描述文件，-bg指定负样本描述文件，-w和-h分别指正样本的宽和高，-num表示正样本的个数。双击create_pos_samples.bat，就会在当前目录下生成一个pos_img.vec文件。

## 5. 开始训练
在train文件夹下新建一个文件夹Haar_xml用以保存训练结果，并在train文件夹下再新建一个txt文件，内容如下
```
opencv_traincascade.exe -data Haar_xml -vec pos.vec -bg neg_img.txt  -numStages 15 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 3450  -numNeg 9000 -w 24 -h 24 -mode ALL
pause
```
参数含义如下表
| 参数名 | 含义 |
| --- | --- |
|-data | 结果输出的位置目录|
|-vec | vec文件 |
|-numPos | 每一级训练时用的正样本数量（建议为所有正样本数量的85%）|
|-numNeg | 每一级训练时用的负样本数量（可以大于所有负样本总数）|
|-numStages | 级连的级数（最大值为20）|
|-stageType | 每一级用什么分类器 |
|-featureType |每一级选用什么特征（可选LBP或HAAR，默认为后者）|
|-w | 图片的宽，必须与之前一致|
|-h | 图片的高，必须与之前一致|
|-bg | 负样本描述文件|
|-minHitRate | 目标每一级的最小真阳率  真阳数/所有正样本数（建议为0.999）|
|-maxFalseAlarmRate | 目标每一级的最大勿检率  假阳数/所有负样本数|
|-maxDepth | 每一级的最大深度|
|-maxWeakCount | 每一级的最大弱分类器数量|
并重命名为traincascade.bat。

双击traincascade.bat文件，如果出现以下内容，恭喜你，训练顺利开始啦！请耐心等待吧~~
![train](E:/study/uestc/senior/DIP/database/ppt/train.png)
训练可中断，再次双击bat文件会继续训练。

## 6. 训练结束
训练结束后会在Haar_xml文件下生成如下文件，其中cascade.xml是最终训练的结果，其他都是中间文件，可删除
![result](E:/study/uestc/senior/DIP/database/ppt/results.png)
python使用自己的分类器检测实例：
```
import cv2
import numpy as np
import tools


def detect(filename):
    # cv2级联分类器CascadeClassifier,xml文件为训练数据
    face_cascade = cv2.CascadeClassifier("XXX/cascade.xml") ###绝对路径
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

img_path = 'XXX/XXX.jpg'
detect(img_path)
```
例子如下
![exsample](E:/study/uestc/senior/DIP/database/ppt/exsample.png)