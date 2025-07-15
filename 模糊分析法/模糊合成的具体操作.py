import numpy as np
def fuzzy_composition(W, R):
    """
    模糊合成运算（加权平均模型）
    :param W: 权重向量 [w1, w2, ..., wm]
    :param R: 评价矩阵 m×n (m因素×n等级)
    :return: 综合评价值 [b1, b2, ..., bn]
    """
    n_levels = R.shape[1]  # 评价等级数量
    B = np.zeros(n_levels)

    # 逐级计算
    for j in range(n_levels):
        # 对每个等级j：计算所有因素i的加权和
        for i in range(len(W)):
            B[j] += W[i] * R[i, j]

    # 归一化处理
    total = np.sum(B)
    if total > 0:
        B_normalized = B / total
    else:
        B_normalized = np.ones(n_levels) / n_levels  # 防除零

    return B_normalized


# 使用示例
W = [0.4, 0.25, 0.2, 0.15]
R = np.array([
    [0.6, 0.3, 0.1, 0.0],
    [0.4, 0.4, 0.2, 0.0],
    [0.3, 0.5, 0.2, 0.0],
    [0.2, 0.5, 0.3, 0.0]
])

B = fuzzy_composition(W, R)
print(f"综合评价值: {B}")  # 输出: [0.430 0.395 0.175 0.000]


###核心就是向量的操作，计算


