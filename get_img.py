# 获取验证码图像样本(1k, only number&character)
import os
from urllib import request
from random import randint
import numpy as np
from PIL import Image
from svmutil import *

def saveSimpleImg(path, number=10):
    """
    获取训练样本并存于本地
    :param path:样本存储路径
    :param number:样本数量
    :return:
    """
    imgurl = "http://localhost:8080/login_j2ee/refpic.jsp?pwdcode="
    if not os.path.exists(path):
        os.mkdir(path)

    pwd = "0123456789QWERTYUIOPASDFGHJKLZXCVBNM"
    for i in range(len(pwd)):
        path2 = path + '/' + pwd[i]
        if not os.path.exists(path2):
            os.mkdir(path2)
        for j in range(number):
            code = pwd[i] * 4
            u = request.urlopen(imgurl + code)
            data = u.read()
            imgpath = path2 + "/%d.png" % (j)
            with open(imgpath, 'wb') as f:
                f.write(data)

def saveTestImg(path, number=50):
    """
    获取测试样本并存于本地
    :param path: 样本存储路径
    :param number: 样本数量
    :return:
    """
    imgurl = "http://localhost:8080/login_j2ee/refpic.jsp?pwdcode="
    if not os.path.exists(path):
        os.mkdir(path)
    pwd = "0123456789QWERTYUIOPASDFGHJKLZXCVBNM"
    for i in range(number):
        code = ''
        for j in range(4):
            code += pwd[randint(0, len(pwd)-1)]
        u = request.urlopen(imgurl + code)
        data = u.read()
        imgpath = path + "/%d.png" % (i)
        with open(imgpath, 'wb') as f:
            f.write(data)

def get_bin_table(threshold=210):
    """
    灰度图二值化
    :param threshold:阈值
    :return: 阈值矩阵
    """
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    return table

def get_crop_imgs(img):
    """
    图像分割
    :param img: 带分割图像
    :return: 分割后字符图像
    """
    child_img_list = []
    for i in range(4):
        x = 10 + i * 30
        y = 0
        child_img = img.crop((x, y, x + 20, y + 50))
        child_img_list.append(child_img)
    return child_img_list

def get_normal_img(img, wigth=10, height=25):
    """
    图像大小归一化
    :param img: 待处理图像
    :param wigth: 归一宽度
    :param height: 归一长度
    :return: 归一后图像
    """
    img_w, img_h = img.size
    h_size = [0, img_h-1]
    w_size = [0, img_w-1]
    img_array = np.array(img, int)
    f = True
    h = 0
    for i in img_array:
        if f:
            if 0 in i:
                h_size[0] = h
                f = not f
        else:
            if 0 not in i:
                h_size[1] = h
                break
        h += 1
    w = 0
    f = True
    for i in img_array.T:
        if f:
            if 0 in i:
                w_size[0] = w
                f = not f
        else:
            if 0 not in i:
                w_size[1] = w
                break
        w += 1
    img = img.crop((w_size[0], h_size[0], w_size[1], h_size[1]))
    img = img.resize([wigth, height])
    return img

def splitSimpleImg(path1, path2):
    """
    对验证码图像原始样本进行二值化、分割提取单字符图像并对图像大小归一化
    :param path1:原始图像样本路径
    :param path2:处理后图像样本路径
    :return:
    """
    if not os.path.exists(path2):
        os.mkdir(path2)
    pwd = "0123456789QWERTYUIOPASDFGHJKLZXCVBNM"
    for i in range(len(pwd)):
        count = 0
        path1_2 = (path1 + '/{0}').format(pwd[i])
        path2_2 = (path2 + '/{0}').format(pwd[i])
        if not os.path.exists(path2_2):
            os.mkdir(path2_2)
        for j in os.listdir(path1_2):
            im = Image.open(path1_2 + '/' + j)
            im = im.convert('L')
            table = get_bin_table()
            out = im.point(table, '1')
            img_list = get_crop_imgs(out)
            for k in img_list:
                k = get_normal_img(k)
                k.save((path2_2 + '/{0}.png').format(str(count)))
                count += 1

def get_feature(img):
    """
    获取图像特征值序列
    :param img:待处理图像
    :return:特征值序列
    """
    wight, height = img.size
    feature_list = []
    img_array = np.array(img, int)
    for h in img_array:
        feature_list.append(wight - np.count_nonzero(h))
    for w in img_array.T:
        feature_list.append(height - np.count_nonzero(w))
    return feature_list

def get_feature_file(imgpath, filepath):
    """
    提取图像特征值生成训练数据集
    :param imgpath:图像集合路径
    :param filepath:文件保存路径
    :return:
    """
    with open(filepath, 'w') as f:
        d = dict()
        k = 0
        for i in os.listdir(imgpath):
            if i not in d:
                d[i] = k
                k += 1
            for j in os.listdir(imgpath + '/' + i):
                path = imgpath + '/' + i + '/' + j
                im = Image.open(path)
                feature_list = get_feature(im)
                trans_row = str(d[i])
                for index, item in enumerate(feature_list):
                    trans_row += ' {}:{}'.format(index+1, item)
                trans_row += '\n'
                f.write(trans_row)

def trains_svm_model(filepath):
    """
    训练并生成model文件
    :param filepath: 训练数据路径
    :return:
    """
    y, x = svm_read_problem(filepath)
    model = svm_train(y, x)
    svm_save_model('model_file', model)

def svm_model_test(testpath):
    """
    使用测试集测试已得模型
    :param testpath:test文件路径
    :return:
    """
    yt, xt = svm_read_problem(testpath)
    model = svm_load_model('model_file')
    p_label, p_acc, p_val = svm_predict(yt, xt, model)
    # cnt = 0
    # for item in p_label:
    #     print('%d' % item, end=',')
    #     cnt += 1
    #     if cnt % 8 == 0:
    #         print('')

def get_trains(trains_path,  number):
    """
    获取测试样本，生成trains文件并使用SVM对训练样本生成对应model文件
    :param trains_path: 训练样本存储路径
    :param number: 训练样本数量
    :return:
    """
    if not os.path.exists(trains_path):
        os.mkdir(trains_path)
    path1 = trains_path + '/' + 'source_img'
    path2 = trains_path + '/' + 'source2_img'
    path3 = trains_path + '/' + 'trains_file.txt'
    saveSimpleImg(path1, number)
    splitSimpleImg(path1, path2)
    get_feature_file(path2, path3)
    trains_svm_model(path3)

def get_test(test_path, number):
    """
    获取测试样本，生成对应test文件通过model文件对SVM进行测试
    :param test_path: 测试样本存储路径
    :param model_path: model文件存储路径
    :param number: 测试样本数量
    :return:
    """
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    path1 = test_path + '/' + 'source_img'
    path2 = test_path + '/' + 'source2_img'
    path3 = test_path + '/' + 'test_file.txt'
    saveSimpleImg(path1, number)
    splitSimpleImg(path1, path2)
    get_feature_file(path2, path3)
    svm_model_test(path3)

def svm_run(trains_path='svm_trains_file', test_path='svm_test_file', trains_num=10, test_num=5):
    """

    :param trains_path:
    :param test_path:
    :param model_path:
    :return:
    """
    get_trains(trains_path, trains_num)
    get_test(test_path, test_num)

svm_run(test_num=10)



