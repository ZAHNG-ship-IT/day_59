import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# ================== 数据预处理 ==================
def data_preprocessing(data, indicators):
    """
    原始数据标准化处理（支持四种指标类型）
    :param data: 原始数据矩阵 (m×n)
    :param indicators: 指标类型列表
        [('type', param1, param2), ...]
        类型: 'pos'(正向), 'neg'(负向), 'range'(区间), 'mid'(中间)
    :return: 标准化后的矩阵
    """
    m, n = data.shape
    norm_data = np.zeros_like(data, dtype=float)

    for j in range(n):
        col = data[:, j]
        min_val = np.min(col)
        max_val = np.max(col)
        type_info = indicators[j]

        if type_info[0] == 'pos':  # 正向指标
            norm_data[:, j] = (col - min_val) / (max_val - min_val + 1e-10)

        elif type_info[0] == 'neg':  # 负向指标
            norm_data[:, j] = (max_val - col) / (max_val - min_val + 1e-10)

        elif type_info[0] == 'range':  # 区间型
            a, b = type_info[1], type_info[2]
            M = max(a - min_val, max_val - b)
            for i in range(m):
                if col[i] < a:
                    norm_data[i, j] = 1 - (a - col[i]) / M
                elif col[i] > b:
                    norm_data[i, j] = 1 - (col[i] - b) / M
                else:
                    norm_data[i, j] = 1

        elif type_info[0] == 'mid':  # 中间型
            c = type_info[1]
            K = max(c - min_val, max_val - c)
            norm_data[:, j] = 1 - np.abs(col - c) / K###
            ###
            ###
            ###
            ###
            ###四种指标类型处理方式












    return norm_data


# ================== 熵权法计算权重 ==================
def entropy_weight(norm_data):####··除以0的优化
    p = norm_data / np.sum(norm_data, axis=0, keepdims=True)

    # 避免log(0)错误
    p_safe = np.where(p == 0, 1e-10, p)

    # 计算信息熵
    m = norm_data.shape[0]
    e = -np.sum(p_safe * np.log(p_safe), axis=0) / np.log(m)

    # 计算权重
    redundancy = 1 - e
    weights = redundancy / np.sum(redundancy)

    return weights, e


# ================== TOPSIS优劣解距离法 ==================
def topsis(norm_data, weights):
    """
    TOPSIS法计算综合得分
    :param norm_data: 标准化后的数据矩阵
    :param weights: 权重向量
    :return: 综合得分 (0~1), 贴近度
    """
    # 加权标准化矩阵
    weighted_matrix = norm_data * weights

    # 确定正负理想解
    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)

    # 计算距离
    dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best) ** 2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst) ** 2, axis=1))

    # 计算贴近度
    closeness = dist_worst / (dist_best + dist_worst)

    # 归一化得分 (0~100)
    scores = 100 * (closeness / np.max(closeness))

    return scores, closeness


# ================== 可视化函数 ==================
def plot_weights(weights, labels, title="熵权法指标权重分布"):
    """绘制权重柱状图"""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(weights)), weights, tick_label=labels)
    plt.title(title, fontsize=15)
    plt.ylabel('权重', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('权重分布.png', dpi=300)
    plt.show()


def plot_scores(scores, labels, title="供应商综合评价得分"):
    """绘制得分雷达图"""
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    # 绘制雷达图
    scores = np.append(scores, scores[0])
    ax.plot(angles, scores, 'o-', linewidth=2)
    ax.fill(angles, scores, alpha=0.25)

    # 设置标签
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    # 设置坐标轴
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.grid(True)

    plt.title(title, fontsize=15, pad=20)
    plt.tight_layout()
    plt.savefig('综合评价雷达图.png', dpi=300)
    plt.show()


def plot_closeness_heatmap(data, closeness, labels, title="供应商优劣解距离热力图"):
    """绘制热力图展示距离关系"""
    plt.figure(figsize=(10, 6))

    # 创建正负理想解列
    ideal_best = np.max(data, axis=0)
    ideal_worst = np.min(data, axis=0)

    # 合并数据
    all_data = np.vstack([ideal_best, ideal_worst, data])
    all_labels = ['正理想解', '负理想解'] + labels

    # 绘制热力图
    sns.heatmap(all_data, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=[f'指标{i + 1}' for i in range(data.shape[1])],
                yticklabels=all_labels)

    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig('优劣解距离热力图.png', dpi=300)
    plt.show()


# =================================================================
#                       2021国赛C题实例应用
# =================================================================
if __name__ == "__main__":
    # 模拟2021国赛C题数据：5个供应商，6项评价指标
    supplier_names = ['供应商A', '供应商B', '供应商C', '供应商D', '供应商E']

    # 原始数据矩阵（根据题目附件数据模拟）
    # 指标顺序：供应量(吨), 合格率(%), 供货周期(天), 价格(万元/吨), 运输距离(km), 环保评分(1-10)
    raw_data = np.array([
        [1200, 95.2, 5, 1.25, 350, 8.5],  # 供应商A
        [950, 97.8, 3, 1.35, 280, 9.2],  # 供应商B
        [1400, 93.5, 7, 1.15, 420, 7.8],  # 供应商C
        [1100, 98.2, 4, 1.40, 310, 8.9],  # 供应商D
        [1300, 96.5, 6, 1.20, 380, 8.0]  # 供应商E
    ])

    # 指标类型定义（根据题目要求）
    indicators = [
        ('pos',),  # 1.供应量（正向：越大越好）
        ('mid', 97),  # 2.合格率（中间：接近97%最好）
        ('neg',),  # 3.供货周期（负向：越小越好）
        ('neg',),  # 4.价格（负向：越小越好）
        ('range', 300, 400),  # 5.运输距离（区间：300-400km最佳）
        ('pos',)  # 6.环保评分（正向：越大越好）
    ]

    print("=" * 60)
    print("2021年全国大学生数学建模竞赛C题：生产企业原材料的订购与运输")
    print("供应商综合评价模型（熵权法+TOPSIS）")
    print("=" * 60)

    # 步骤1：数据标准化
    norm_data = data_preprocessing(raw_data, indicators)
    print("\n>>> 标准化后的数据矩阵：")
    print(pd.DataFrame(norm_data, index=supplier_names,
                       columns=['供应量', '合格率', '供货周期', '价格', '运输距离', '环保']))

    # 步骤2：熵权法计算权重
    weights, entropies = entropy_weight(norm_data)
    print("\n>>> 信息熵值：", np.round(entropies, 4))
    print(">>> 指标权重：", np.round(weights, 4))

    # 可视化权重分布
    plot_weights(weights,
                 ['供应量', '合格率', '供货周期', '价格', '运输距离', '环保'],
                 "原材料供应商评价指标权重")

    # 步骤3：TOPSIS综合评价
    scores, closeness = topsis(norm_data, weights)

    # 按得分排序
    ranked_suppliers = sorted(zip(supplier_names, scores, closeness),
                              key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("供应商综合评价排名：")
    print("排名\t供应商\t综合得分\t贴近度")
    for i, (name, score, cls) in enumerate(ranked_suppliers, 1):
        print(f"{i}\t{name}\t{score:.2f}\t\t{cls:.4f}")
    print("=" * 60)

    # 可视化得分结果
    plot_scores([s for _, s, _ in ranked_suppliers],
                [name for name, _, _ in ranked_suppliers],
                "供应商综合评价雷达图")

    # 优劣解距离热力图
    plot_closeness_heatmap(norm_data, closeness, supplier_names)

    # 生成最终报告
    print("\n>>> 建模结论：")
    print(f"1. 最优供应商：{ranked_suppliers[0][0]}（得分 {ranked_suppliers[0][1]:.2f}）")
    print(
        f"2. 关键指标权重：合格率({weights[1] * 100:.1f}%) > 价格({weights[3] * 100:.1f}%) > 供货周期({weights[2] * 100:.1f}%)")
    print("3. 建议签订采购合同比例：")
    total_score = sum(score for _, score, _ in ranked_suppliers)
    for name, score, _ in ranked_suppliers:
        percent = (score / total_score) * 100
        print(f"   - {name}: {percent:.1f}%")
