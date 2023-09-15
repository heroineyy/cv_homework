import cv2
import numpy as np


# 自定义函数实现直方图均衡
def custom_histogram_equalization(image):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算直方图
    hist, bins = np.histogram(gray_image.flatten(), bins=256, range=(0, 256))

    # 计算累积分布函数（CDF）
    cdf = hist.cumsum()

    # 归一化CDF
    cdf_normalized = cdf * hist.max() / cdf.max()

    # 使用CDF进行均衡化
    equalized_image = np.interp(gray_image, bins[:-1], cdf_normalized)

    return equalized_image.astype(np.uint8)


# 读取图像
image = cv2.imread(r'/horses.jpg')

# 应用自定义直方图均衡
equalized_image = custom_histogram_equalization(image)

# 显示原始图像和均衡化后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)

# 等待用户按键并关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
