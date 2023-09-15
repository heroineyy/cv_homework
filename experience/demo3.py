import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def find_baboon_nose_thresholds(image_path):
    # 读取输入图片
    image = cv2.imread(image_path)

    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算灰度直方图
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    histogram = histogram.flatten()

    # 使用SciPy的find_peaks函数找到直方图中的峰值
    peaks, _ = find_peaks(histogram, distance=30)  # 调整distance以适应图像

    # 如果找到了两个峰值，则将它们作为双阈值返回
    if len(peaks) >= 2:
        lower_threshold = int(peaks[0])
        upper_threshold = int(peaks[1])
        return lower_threshold, upper_threshold
    else:
        # 找不到两个峰值，返回默认值或者根据需要进行处理
        return None

# 使用示例
input_image_path = r'D:\deeplearning\cv_homework\data\baboon.png'
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
plt.plot(histogram)
plt.title('Gray Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
