import cv2
import numpy as np
import matplotlib.pyplot as plt


def restore1(u, sigma, v, k):

    m = len(u)
    n = len(v)
    a = np.zeros((m, n))
    a = np.dot(u[:, :k], np.diag(sigma[:k])).dot(v[:k, :])
    # s1 =  np.size(u[:, :k])
    # s1+= np.size(np.diag(sigma[:k]))
    # s1+= np.size(np.diag(v[:k, :]))
    # s2 = np.size(a)
    # print("压缩率：",s1/s2)
    a[a < 0] = 0
    a[a > 255] = 255
    return np.rint(a).astype('uint8')


def SVD(frame, K=10):
    a = np.array(frame)
    # 由于是彩色图像，所以3通道。a的最内层数组为三个数，分别表示RGB，用来表示一个像素
    u_r, sigma_r, v_r = np.linalg.svd(a[:, :, 0])
    u_g, sigma_g, v_g = np.linalg.svd(a[:, :, 1])
    u_b, sigma_b, v_b = np.linalg.svd(a[:, :, 2])

    R = restore1(u_r, sigma_r, v_r, K)
    G = restore1(u_g, sigma_g, v_g, K)
    B = restore1(u_b, sigma_b, v_b, K)
    I = np.stack((R, G, B), axis=2)
    return I


if __name__ == "__main__":
    img = cv2.imread("./imageCompressionGray.JPG", 0)
    # I = SVD(img, 40)
    # plt.imshow(I)
    [u, sigma, v] = np.linalg.svd(img)
    # cv2.imwrite("out.bmp", I)
    x = range(len(sigma))
    plt.plot(x, sigma)
    plt.xlabel('rank-k')
    plt.ylabel('sigma-k')
    # plt.show()
    plt.savefig("matplotlib.png")
