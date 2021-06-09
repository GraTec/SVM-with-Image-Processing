import cv2
import numpy as np
import time
from sklearn import svm

# from SVM import SVM

filePath = ['./cifar-10-batches-py/data_batch_'+str(x) for x in range(1, 6)]


def loadData():  # 加载数据
    x_train, y_train = [], []
    for file in filePath:
        dict = unpickle(file)
        data = dict[b'data']
        labels = dict[b'labels']  # 3-cats
        for i in range(len(labels)):
            m = data[i, :]
            x_train.append(m)
            if labels[i] == 3:  # cats
                y_train.append(1)
            else:
                y_train.append(-1)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test, y_test = [], []
    dict = unpickle('./cifar-10-batches-py/test_batch')
    data = dict[b'data']
    labels = dict[b'labels']  # 3-cats
    for i in range(len(labels)):
        m = data[i, :]
        x_test.append(m)
        if labels[i] == 3:  # cats
            y_test.append(1)
        else:
            y_test.append(-1)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test


def unpickle(file):  # 读取数据集
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def imgGray(imgData):  # 将图片转化为灰度图
    newData = []
    for i in range(len(imgData)):
        img = imgData[i]  # 0-1023,1024-2047,2048-3071
        img_R = img[0:1024].reshape((32, 32))
        img_G = img[1024:2048].reshape((32, 32))
        img_B = img[2048:3072].reshape((32, 32))
        imgColor = np.stack((img_R, img_G, img_B), axis=2)
        imgGray = cv2.cvtColor(imgColor, cv2.COLOR_RGB2GRAY)
        newData.append(imgGray)
    return np.array(newData)


def computeGradient(img):  # 计算单个图片的梯度幅值和方向图
    # 计算x和y方向的梯度
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # 计算合梯度的幅值和方向（角度）
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = angle % 180
    return mag, angle


def angleJudge(angle):  # 判断每个angle对应的下标
    return int(angle/20), (int(angle/20)+1) % 9


def imgProcess(imgData):  # 将数据处理成HOG的形式
    newImgData = []
    for i in range(len(imgData)):
        img = imgData[i]  # 单张图片
        mag, angle = computeGradient(img)
        # print(mag)
        # 每个图片为32x32, 以8x8为一个cell, 一个cell储存一个长度为9的直方图
        cell = np.zeros((4, 4, 9))
        for x in range(32):  # 8x->8(x+1)-1
            for y in range(32):
                sub, sub1 = angleJudge(angle[x, y])
                cell[int(x/8), int(y/8), sub] += mag[x, y]/2
                cell[int(x/8), int(y/8), sub1] += mag[x, y]/2
        # 现在进行Block归一化
        HOG = np.array([])
        for x in range(3):
            for y in range(3):
                block = np.concatenate(
                    (cell[x, y], cell[x, y+1], cell[x+1, y], cell[x+1, y+1]))
                if np.linalg.norm(block) != 0:
                    block = block/np.linalg.norm(block)
                HOG = np.concatenate((HOG, block))
        newImgData.append(HOG)
        # print(i, end='\r')
    return np.array(newImgData)


def compare(y_test, y_predict):  # 计算误差
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] != y_predict[i]:
            correct += 1
    return 1-correct/len(y_test)


def initialize():
    x_train, y_train, x_test, y_test = loadData()
    x_train = imgGray(x_train)
    x_test = imgGray(x_test)
    x_train = imgProcess(x_train)
    x_test = imgProcess(x_test)
    np.save('x_train.npy', x_train)
    np.save('x_test.npy', x_test)


if __name__ == '__main__':
    # initialize()

    x_train, y_train, x_test, y_test = loadData()

    x_train = np.load('x_train.npy')  # 50,000个数据，以10,000为一组
    x_test = np.load('x_test.npy')

    # 对每一组数据计算其误差和运行时间
    error = []
    runTime = []
    for i in range(0, 5):
        # i*10000-(i+1)*10000
        # print('Initialization finished.')
        print('Running part '+str(i)+' of the dataset......')
        start = time.process_time()
        clf = svm.SVC(C=1.0, kernel='rbf', gamma='scale',
                      decision_function_shape='ovr')
        clf.fit(x_train[10000*i:(i+1)*10000], y_train[10000*i:(i+1)*10000])
        end = time.process_time()
        # 预测测试集
        y_predict = clf.predict(x_test)
        cnt = 0
        error.append(compare(y_test, y_predict))
        # error.append(clf.score(x_test, y_test))
        runTime.append(end-start)
        # print(clf.score(x_test, y_test))
    print(runTime, error)

# [9.46283037, 9.386277245999995, 9.179519941999999, 9.378836269999994, 9.39051176000001] [0.9003, 0.9002, 0.9009, 0.9003, 0.9004]
