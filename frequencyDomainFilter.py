import matplotlib.pyplot as plt
import numpy as np
import cv2

# 频域滤波器类
class FDF:

    # 高通滤波
    def highPassFilter(self,image,HDistance):
        # 使用numpy带的fft库完成从空间域到频率域的转换
        f = np.fft.fft2(image)
        # 将零频点移到频谱的中间，中心化
        fshift = np.fft.fftshift(f)
        # 取绝对值：将复数变化成实数
        # 取对数的目的为了将数据变化到0-255
        s1 = np.log(np.abs(fshift))
        # 通过不同的滤频半径distance来过滤
        def make_transform_matrix(HDistance):
            transfor_matrix = np.zeros(image.shape)
            center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
            for i in range(transfor_matrix.shape[0]):
                for j in range(transfor_matrix.shape[1]):
                    def cal_distance(pa, pb):
                        from math import sqrt
                        dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                        return dis
                    dis = cal_distance(center_point, (i, j))
                    if dis <= HDistance:
                        transfor_matrix[i, j] = 0
                    else:
                        transfor_matrix[i, j] = 1
            return transfor_matrix
        d_matrix = make_transform_matrix(HDistance)
        # 逆傅里叶变换转换回空间域
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
        return new_img

    # 低通滤波，和高通滤波非常类似，只不过二者通过的波正好是相反的，因此无需重复解释
    def lowPassFilter(self,image,LDistance):
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        s1 = np.log(np.abs(fshift))
        def make_transform_matrix(LDistance):
            transfor_matrix = np.zeros(image.shape)
            center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
            for i in range(transfor_matrix.shape[0]):
                for j in range(transfor_matrix.shape[1]):
                    def cal_distance(pa, pb):
                        from math import sqrt
                        dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                        return dis
                    dis = cal_distance(center_point, (i, j))
                    if dis <= LDistance:
                        transfor_matrix[i, j] = 1
                    else:
                        transfor_matrix[i, j] = 0
            return transfor_matrix
        d_matrix = make_transform_matrix(LDistance)
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
        return new_img

if __name__ == '__main__':
    # 原图像
    originImage = cv2.imread('origin.jpg',0)
    plt.subplot(1, 3, 1)
    plt.title("origin")
    plt.imshow(originImage,'gray')
    # 频率域滤波算法参数
    HDistance = input("please define highDistance：")
    LDistance = input("please define lowDistance：")
    # 构造频域滤波器对象
    fdf = FDF()
    # 输出高通
    dest1 = fdf.highPassFilter(originImage,int(HDistance))
    plt.subplot(1, 3, 2)
    plt.title('HDistance=' + HDistance)
    plt.imshow(dest1,'gray')
    # 输出低通
    dest2 = fdf.lowPassFilter(originImage,int(LDistance))
    plt.subplot(1, 3, 3)
    plt.title('LDistance=' + LDistance)
    plt.imshow(dest2,'gray')

    plt.show()
