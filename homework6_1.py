import numpy as np
import cv2

# 加载特征点
matches = np.loadtxt('matches.txt', delimiter=',')
pts1 = matches[:, :2]
pts2 = matches[:, 2:]


def estimate_fundamental_matrix_ransac(pts1, pts2, threshold, max_iterations):
    best_F = None
    best_inliers = []

    for _ in range(max_iterations):
        # 随机选择8个点用于基础矩阵估计
        random_indices = np.random.choice(len(pts1), 8, replace=False)
        sampled_pts1 = pts1[random_indices]
        sampled_pts2 = pts2[random_indices]

        # 用随机选择的点估计基础矩阵
        F, mask = cv2.findFundamentalMat(sampled_pts1, sampled_pts2, cv2.FM_RANSAC)

        # 计算内点数量
        inliers = np.where(mask.ravel() == 1)[0]

        # 如果内点数量大于当前最佳，则更新最佳结果
        if len(inliers) > len(best_inliers):
            best_F = F
            best_inliers = inliers

    # 用所有内点重新估计一次基础矩阵
    best_pts1 = pts1[best_inliers]
    best_pts2 = pts2[best_inliers]
    best_F, _ = cv2.findFundamentalMat(best_pts1, best_pts2, cv2.FM_8POINT)

    return best_F, best_inliers


thresholds = [1.0, 1.5, 2.0]  # 不同的阈值
max_iterations = 1000

for threshold in thresholds:
    F, inliers = estimate_fundamental_matrix_ransac(pts1, pts2, threshold, max_iterations)
    print(f"Threshold: {threshold}, Number of inliers: {len(inliers)}")

    # 在这里可以根据需要进一步处理基础矩阵 F
