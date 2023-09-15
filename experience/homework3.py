import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal import find_peaks

def smooth_histogram(histogram, kernel_size=5):
    # 创建高斯滤波器内核
    kernel = np.exp(-np.linspace(-1, 1, kernel_size)**2 / 2)
    kernel /= kernel.sum()

    # 对直方图应用滤波器
    smoothed_histogram = convolve(histogram, kernel, mode='same')

    return smoothed_histogram

def find_baboon_nose_thresholds(image_path):
    # 读取输入图片
    image = cv2.imread(image_path)

    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算灰度直方图
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    histogram = histogram.flatten()

    # 使用滤波器平滑直方图
    smoothed_histogram = smooth_histogram(histogram)

    # 使用SciPy的find_peaks函数找到平滑后的直方图中的峰值
    peaks, _ = find_peaks(smoothed_histogram, distance=30)  # 调整distance以适应图像

    # 如果找到了两个峰值，则将它们作为双阈值返回
    if len(peaks) >= 2:
        lower_threshold = int(peaks[0])
        upper_threshold = int(peaks[1])
        return lower_threshold, upper_threshold
    else:
        # 找不到两个峰值，返回默认值或者根据需要进行处理
        return None

# 使用示例
input_image_path = r'/data/baboon.png'
thresholds = find_baboon_nose_thresholds(input_image_path)

if thresholds is not None:
    lower_threshold, upper_threshold = thresholds
    print(f"Lower Threshold: {lower_threshold}")
    print(f"Upper Threshold: {upper_threshold}")
else:
    print("Unable to find suitable thresholds.")

# 可视化直方图
image = cv2.imread(input_image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
histogram = histogram.flatten()

# 平滑后的直方图
smoothed_histogram = smooth_histogram(histogram)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(histogram)
plt.title('Original Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.plot(smoothed_histogram)
plt.title('Smoothed Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
