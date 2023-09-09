import cv2
import numpy as np
import matplotlib.pyplot as plt


# 自定义函数计算灰度直方图
def calculate_custom_histogram(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = np.zeros(256, dtype=np.int64)

    for pixel_value in np.ravel(gray_image):
        hist[pixel_value] += 1

    return hist


# 使用OpenCV计算灰度直方图
def calculate_opencv_histogram(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    return hist


# 读取图像
image = cv2.imread(r'D:\deeplearning\cv_homework\horses.jpg')

# 计算灰度直方图
custom_histogram = calculate_custom_histogram(image)
opencv_histogram = calculate_opencv_histogram(image)

# 显示灰度直方图
plt.figure(figsize=(12, 6))

# 自定义灰度直方图
plt.subplot(121)
plt.title('Custom Histogram')
plt.bar(range(256), custom_histogram, width=1.0, color='b')

# 使用OpenCV计算的灰度直方图
plt.subplot(122)
plt.title('OpenCV Histogram')
plt.plot(opencv_histogram, color='r')

plt.show()
