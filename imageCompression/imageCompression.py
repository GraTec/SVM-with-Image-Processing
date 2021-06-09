import cv2
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt


def imgCompression(u, sigma, v, k):
    m = len(u)
    n = len(v)
    a = np.dot(u[:, :k], np.diag(sigma[:k])).dot(v[:k, :])
    # s1 =  np.size(u[:, :k])
    # s1+= np.size(np.diag(sigma[:k]))
    # s1+= np.size(np.diag(v[:k, :]))
    # s2 = np.size(a)
    # print("压缩率：",s1/s2)
    ratio = (m+n)*k/(m*n)  # 压缩率
    a[a < 0] = 0
    a[a > 255] = 255
    return ratio, np.rint(a).astype('uint8')


def SVDColor(img):
    # 由于是彩色图像，所以3通道。a的最内层数组为三个数，分别表示RGB，用来表示一个像素
    u_r, sigma_r, v_r = np.linalg.svd(img[:, :, 0])
    u_g, sigma_g, v_g = np.linalg.svd(img[:, :, 1])
    u_b, sigma_b, v_b = np.linalg.svd(img[:, :, 2])
    # maxK = len(sigma_r)  # 奇异值的数量，每10%为一个level
    ratio, kValue = [], [10, 50, 100, 200, 400, 800]
    for k in kValue:
        r, imgR = imgCompression(u_r, sigma_r, v_r, k)
        r, imgG = imgCompression(u_g, sigma_g, v_g, k)
        r, imgB = imgCompression(u_b, sigma_b, v_b, k)
        imgNew = np.stack((imgR, imgG, imgB), axis=2)
        ratio.append(r)
        cv2.imwrite("img"+str(k)+".jpg", imgNew)
    print(ratio, kValue)


def SVDGray(img):
    [u, sigma, v] = np.linalg.svd(np.array(img))  # 进行SVD分解
    # cv2.imwrite("out.bmp", I)
    maxK = len(sigma)  # 奇异值的数量，每10%为一个level
    ratio, kValue = [], []
    for i in range(10):
        k = int(maxK/10*(i+1))
        kValue.append(k)
        r, newImg = imgCompression(u, sigma, v, k)
        ratio.append(r)
        cv2.imwrite("img"+str(k)+".jpg", newImg)
    print(ratio, kValue)


if __name__ == "__main__":
    # img = cv2.imread("./imageCompressionGray.JPG", 0)  # 载入图片(灰度)
    img = cv2.imread("./imageCompression.JPG")  # 载入图片(彩色)
    # m, n = len(img), len(img[0])
    # zero = np.zeros((m, n))
    # print(m, n)
    # cv2.imwrite("imgR.jpg", np.stack((img[:, :, 0], zero, zero), axis=2))
    # cv2.imwrite("imgG.jpg", np.stack((zero, img[:, :, 1], zero), axis=2))
    # cv2.imwrite("imgB.jpg", np.stack((zero, zero, img[:, :, 2]), axis=2))
    # SVDGray(img)
    SVDColor(img)
