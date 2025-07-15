import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MultiLevelFuzzyEvaluation:
    def __init__(self, first_level_factors, second_level_factors, evaluations):
        """
        初始化二级模糊评价模型
        :param first_level_factors: 一级指标列表
        :param second_level_factors: 二级指标字典 {一级指标: [二级指标列表]}
        :param evaluations: 评价等级列表
        """
        self.first_level_factors = first_level_factors
        self.second_level_factors = second_level_factors
        self.evaluations = evaluations

    def evaluate(self, R_matrices, first_weights, second_weights):
        """
        执行二级模糊评价
        :param R_matrices: 评价矩阵字典 {一级指标: 二级评价矩阵}
        :param first_weights: 一级指标权重
        :param second_weights: 二级指标权重字典 {一级指标: 权重列表}
        :return: 最终评价结果
        """
        # 第一步：对每个一级指标进行二级模糊评价
        first_level_results = {}
        for factor in self.first_level_factors:
            # 获取该一级指标下的二级指标权重
            weights = second_weights[factor]
            # 获取评价矩阵
            R = R_matrices[factor]
            # 进行模糊合成
            B = self._fuzzy_composition(weights, R)
            first_level_results[factor] = B

        # 第二步：构建一级评价矩阵
        R_first = np.array([first_level_results[factor] for factor in self.first_level_factors])

        # 第三步：进行一级模糊评价
        final_B = self._fuzzy_composition(first_weights, R_first)

        # 综合评价结果
        eval_index = np.argmax(final_B)
        final_evaluation = self.evaluations[eval_index]

        # 计算综合得分
        score_standards = [95, 85, 75, 60]
        composite_score = sum(final_B * np.array(score_standards))

        return {
            "first_level_results": first_level_results,
            "final_membership": final_B,
            "final_evaluation": final_evaluation,
            "composite_score": composite_score
        }

    def _fuzzy_composition(self, weights, R):
        """
        模糊合成运算
        :param weights: 权重向量
        :param R: 评价矩阵
        :return: 归一化的评价值
        """
        n_levels = R.shape[1]
        B = np.zeros(n_levels)

        # 逐级计算
        for j in range(n_levels):
            for i in range(len(weights)):
                B[j] += weights[i] * R[i, j]

        # 归一化处理
        total = np.sum(B)
        if total > 0:
            return B / total
        else:
            return np.ones(n_levels) / n_levels

    def visualize_results(self, results):
        """
        可视化评价结果
        :param results: 评价结果字典
        """
        # 创建图表
        fig = plt.figure(figsize=(16, 10))

        # 一级指标评价结果 - 雷达图
        ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2, projection='polar')
        self._plot_radar(ax1, results)

        # 最终评价结果 - 柱状图
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        self._plot_final_bar(ax2, results)

        # 一级指标对比 - 热力图
        ax3 = plt.subplot2grid((2, 2), (1, 1))
        self._plot_heatmap(ax3, results)

        # 添加综合结论
        final_score = results['composite_score']
        final_eval = results['final_evaluation']
        plt.figtext(0.5, 0.01,
                    f"综合评估：{final_eval} | 综合得分：{final_score:.1f}分",
                    ha="center", fontsize=14, bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5})

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig('多层次模糊评价结果.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_radar(self, ax, results):
        """绘制一级指标雷达图"""
        n_factors = len(self.first_level_factors)
        angles = np.linspace(0, 2 * np.pi, n_factors, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        # 准备雷达图数据
        radar_data = []
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.evaluations)))

        for i, level in enumerate(self.evaluations):
            level_data = []
            for factor in self.first_level_factors:
                # 获取该一级指标在该等级的评价
                level_data.append(results['first_level_results'][factor][i])

            # 闭合多边形
            level_data.append(level_data[0])
            radar_data.append(level_data)

            # 绘制雷达图
            ax.plot(angles, radar_data[i], 'o-', color=colors[i], linewidth=2, label=f"{level}隶属度")
            ax.fill(angles, radar_data[i], color=colors[i], alpha=0.1)

        # 设置角度刻度
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), self.first_level_factors)

        # 设置径向刻度
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

        plt.title('一级指标在各评价等级的隶属度分布', fontsize=14, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    def _plot_final_bar(self, ax, results):
        """绘制最终评价柱状图"""
        final_B = results['final_membership']
        colors = ['#4CAF50', '#8BC34A', '#FFC107', '#F44336']

        # 柱状图
        bars = ax.bar(self.evaluations, final_B, color=colors)
        ax.set_title('最终综合评价结果', fontsize=14)
        ax.set_ylabel('隶属度', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f'{height:.3f}', ha='center', fontsize=10)

        # 添加综合评分
        ax.text(0.05, 0.95, f"综合得分: {results['composite_score']:.1f}",
                transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='yellow', alpha=0.5))

    def _plot_heatmap(self, ax, results):
        """绘制一级指标热力图"""
        # 准备数据
        data = []
        for factor in self.first_level_factors:
            data.append(results['first_level_results'][factor])

        # 转置数据：行=一级指标，列=评价等级
        data = np.array(data).T

        # 绘制热力图
        im = ax.imshow(data, cmap='YlGnBu', aspect='auto')

        # 设置坐标轴
        ax.set_xticks(np.arange(len(self.first_level_factors)))
        ax.set_xticklabels(self.first_level_factors)
        ax.set_yticks(np.arange(len(self.evaluations)))
        ax.set_yticklabels(self.evaluations)

        # 添加数值
        for i in range(len(self.evaluations)):
            for j in range(len(self.first_level_factors)):
                text = ax.text(j, i, f"{data[i, j]:.3f}",
                               ha="center", va="center", color="black")

        plt.title('各一级指标评价分布', fontsize=14)
        plt.colorbar(im, ax=ax, label='隶属度')
        plt.tight_layout()


# ====================== 实例应用 ======================
if __name__ == "__main__":
    print("=== 高校教师教学质量多层次模糊评价系统 ===")

    # 1. 定义评价体系
    first_level_factors = ["教学态度", "教学内容", "教学方法", "教学效果"]
    second_level_factors = {
        "教学态度": ["备课充分", "认真负责", "师德师风"],
        "教学内容": ["内容深度", "内容广度", "前沿性"],
        "教学方法": ["方法多样", "互动交流", "技术应用"],
        "教学效果": ["知识掌握", "能力提升", "满意度"]
    }
    evaluations = ["优秀", "良好", "合格", "不合格"]

    # 2. 设置权重
    first_weights = [0.20, 0.30, 0.25, 0.25]
    second_weights = {
        "教学态度": [0.30, 0.40, 0.30],
        "教学内容": [0.40, 0.30, 0.30],
        "教学方法": [0.30, 0.40, 0.30],
        "教学效果": [0.35, 0.35, 0.30]
    }

    # 3. 设置评价矩阵
    R_matrices = {
        "教学态度": np.array([
            [0.6, 0.3, 0.1, 0.0],  # 备课充分
            [0.5, 0.4, 0.1, 0.0],  # 认真负责
            [0.7, 0.2, 0.1, 0.0]  # 师德师风
        ]),
        "教学内容": np.array([
            [0.5, 0.3, 0.2, 0.0],  # 内容深度
            [0.4, 0.4, 0.2, 0.0],  # 内容广度
            [0.3, 0.5, 0.2, 0.0]  # 前沿性
        ]),
        "教学方法": np.array([
            [0.4, 0.4, 0.2, 0.0],  # 方法多样
            [0.5, 0.3, 0.2, 0.0],  # 互动交流
            [0.6, 0.3, 0.1, 0.0]  # 技术应用
        ]),
        "教学效果": np.array([
            [0.4, 0.3, 0.2, 0.1],  # 知识掌握
            [0.3, 0.4, 0.2, 0.1],  # 能力提升
            [0.5, 0.3, 0.1, 0.1]  # 满意度
        ])
    }

    # 4. 创建模型并执行评价
    model = MultiLevelFuzzyEvaluation(first_level_factors, second_level_factors, evaluations)
    results = model.evaluate(R_matrices, first_weights, second_weights)

    # 5. 输出结果
    print("\n=== 一级指标评价结果 ===")
    for factor in first_level_factors:
        print(f"{factor}: {np.round(results['first_level_results'][factor], 3)}")

    print("\n=== 最终综合评价 ===")
    print(f"隶属度分布: {np.round(results['final_membership'], 3)}")
    print(f"最终评价: {results['final_evaluation']}")
    print(f"综合得分: {results['composite_score']:.1f}分")

    # 6. 可视化结果
    model.visualize_results(results)

    # 7. 结果分析
    print("\n=== 教学质量分析报告 ===")
    print("1. 教学态度方面表现突出，师德师风获得高度认可")
    print("2. 教学效果方面需要加强，特别是学生能力提升维度")
    print("3. 建议：")
    print("   - 增加课堂互动活动，提高学生参与度")
    print("   - 更新教学内容，融入更多前沿知识")
    print("   - 定期收集学生反馈，针对性改进教学")
