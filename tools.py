def get_all_files(img_path, mode=0):
    import os
    Filelist = []
    for home, dirs, files in os.walk(img_path):
        for filename in files:
            if mode == 0:
            # 文件名列表，只包含文件名
                Filelist.append(filename)
            if mode == 1:
                # 文件名列表，包含完整路径
                Filelist.append(os.path.join(home,filename)) 
    return Filelist

def read(img_path):
    import cv2
    import numpy as np
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
    return img

def save(img, save_path, type_='.bmp',):
    import cv2
    cv2.imencode(type_, img)[1].tofile(save_path)

def copy(img_path, save_path):
    import shutil
    img_list = get_all_files(img_path, mode=0)
    for img_path in img_list:
        # save_path 里要用 /
        shutil.copy(img_path, save_path)

def rename(img_path, type_='.jpg'):
    # img_path 里要用 /
    import os
    img_list = get_all_files(img_path, mode=1)
    for i in range(len(img_list)):
        path = img_path + '/' + str(i) + type_
        os.rename(img_list[i], path) 

def resize(img, dim):
    import cv2
    img_resized = cv2.resize(img, dim)
    return img_resized

def grayImg(img_path,type_):
    import cv2
    #读取图片
    img = read(img_path)
    GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #保存灰度后的新图片
    save(GrayImage,img_path,type_)

def create_pos_txt(path):
    i_new = []
    with open(path, 'r') as f:
        for i in f.readlines():
            i_new.append('./pos_img/'+i[:-1]+' 1 0 0 24 24\n') ### 改成对应的参数
    with open(path, 'w') as f:
        f.writelines(i_new)

def create_neg_txt(path):
    i_new = []
    with open(path, 'r') as f:
        for i in f.readlines():
            i_new.append('./neg_img/'+i) ### 改成对应的参数
    with open(path, 'w') as f:
        f.writelines(i_new)


if __name__ == '__main__':
    # path = 'E:/study/uestc/senior/DIP/program/face_detection/train/neg_img'
    # img_list = get_all_files(path, mode=1)
    # for img_path in img_list:
    #     img = read(img_path)
    #     resized = resize(img, (50,50))
    #     save(resized, img_path, type_='.jpg')
    # print("Done!")

    # rename(path, type_='.jpg')

    # for img_path in img_list:
    #     grayImg(img_path, type_='.jpg')

    # txt_path = 'E:/study/uestc/senior/DIP/program/face_detection/train/pos_img.txt'
    # create_pos_txt(txt_path)

    txt_path = 'E:/study/uestc/senior/DIP/program/face_detection/train/neg_img.txt'
    create_neg_txt(txt_path)