# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.font_manager import FontProperties
#
# # 设置中文显示
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
#
# class FuzzyEvaluation:
#     def __init__(self, factors, evaluations):
#         """
#         初始化模糊综合评价模型
#         :param factors: 评价因素列表
#         :param evaluations: 评价等级列表
#         """
#         self.factors = factors
#         self.evaluations = evaluations
#         self.weights = None
#         self.R_matrix = None
#
#     def set_weights(self, weights):
#         """设置权重向量"""
#         if abs(sum(weights) - 1.0) > 0.01:
#             raise ValueError("权重总和必须为1")
#         if len(weights) != len(self.factors):
#             raise ValueError("权重数量与因素数量不匹配")
#         self.weights = np.array(weights)
#
#     def set_evaluation_matrix(self, R):
#         """设置单因素评价矩阵"""
#         if R.shape != (len(self.factors), len(self.evaluations)):
#             raise ValueError("评价矩阵维度错误")
#         self.R_matrix = R
#
#     def evaluate(self, show_result=True):
#         """执行模糊综合评价"""
#         if self.weights is None or self.R_matrix is None:
#             raise ValueError("权重或评价矩阵未设置")
#
#         # 模糊合成运算
#         B = np.zeros(len(self.evaluations))
#         for j in range(len(self.evaluations)):
#             B[j] = sum(self.weights[i] * self.R_matrix[i, j]
#                        for i in range(len(self.factors)))
#
#         # 归一化处理
#         B_normalized = B / np.sum(B)
# ##消除量纲
#         # 综合评价结果
#         eval_index = np.argmax(B_normalized)
#         final_evaluation = self.evaluations[eval_index]
#
#         # 计算综合得分（假设评分标准）
#         score_standards = [95, 85, 70, 55]  # 优秀/良好/合格/不合格对应分数
#         composite_score = sum(B_normalized * np.array(score_standards))
#
#         if show_result:
#             self._visualize_results(B_normalized, final_evaluation, composite_score)
#
#         return {
#             'membership_degrees': B_normalized,
#             'final_evaluation': final_evaluation,
#             'composite_score': composite_score
#         }
#
#     def _visualize_results(self, B, final_eval, score):
#         """可视化评价结果"""
#         # 创建图表
#         fig, ax = plt.subplots(1, 2, figsize=(14, 6))
#
#         # 子图1：隶属度分布柱状图
#         colors = ['#4CAF50', '#8BC34A', '#FFC107', '#F44336']
#         ax[0].bar(self.evaluations, B, color=colors)
#         ax[0].set_title('各评价等级隶属度分布', fontsize=14)
#         ax[0].set_ylabel('隶属度', fontsize=12)
#         ax[0].grid(axis='y', linestyle='--', alpha=0.7)
#
#         # 添加数值标签
#         for i, v in enumerate(B):
#             ax[0].text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)
#
#         # 子图2：雷达图展示各因素评价
#         angles = np.linspace(0, 2 * np.pi, len(self.factors), endpoint=False).tolist()
#         angles += angles[:1]  # 闭合
#
#         # # 扩展矩阵用于绘图
#         plot_data = self.R_matrix.T
#         plot_data = np.concatenate((plot_data, plot_data[:1]))
# ##绘制雷达图维度出错了，一下有代码
#         # radar_data = []
#         # for i, level in enumerate(self.evaluations):
#         #     # 获取该等级下各因素的评价
#         #     level_data = self.R_matrix[:, i].tolist()
#         #     level_data.append(level_data[0])  # 闭合多边形
#         #     radar_data.append(level_data)
#         #
#         # 创建雷达图
#         ax[1] = plt.subplot(122, polar=True)
#         for i, level in enumerate(self.evaluations):
#             ax[1].plot(angles, plot_data[i], 'o-', label=level, linewidth=2)
#
#         # 设置角度刻度
#         ax[1].set_theta_offset(np.pi / 2)
#         ax[1].set_theta_direction(-1)
#         ax[1].set_thetagrids(np.degrees(angles[:-1]), self.factors)
#
#         # 设置径向刻度
#         ax[1].set_rlabel_position(0)
#         ax[1].set_ylim(0, 1)
#
#         plt.title('各因素在不同评价等级的分布', fontsize=14, pad=20)
#         plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
#
#         # 添加综合结论
#         plt.figtext(0.5, 0.05,
#                     f"综合评估结果：{final_eval} | 综合得分：{score:.1f}分",
#                     ha="center", fontsize=14, bbox={"facecolor": "orange", "alpha": 0.3, "pad": 5})
#
#         plt.tight_layout()
#         plt.subplots_adjust(bottom=0.2)
#         plt.savefig('模糊综合评价结果.png', dpi=300, bbox_inches='tight')
#         plt.show()
#
#
# # ============================= 实例应用 =============================
# if __name__ == "__main__":
#     print("=== 员工绩效模糊综合评价系统 ===")
#
#     # 1. 初始化评价体系
#     factors = ['工作业绩', '工作态度', '团队协作', '学习能力']
#     evaluations = ['优秀', '良好', '合格', '不合格']
#     model = FuzzyEvaluation(factors, evaluations)
#
#     # 2. 设置权重 (使用AHP方法确定)
#     model.set_weights([0.40, 0.25, 0.20, 0.15])
#
#     # 3. 设置评价矩阵 (来自专家评分)
#     # 5位专家对员工"张三"的评价统计
#     R = np.array([
#         [0.6, 0.3, 0.1, 0.0],  # 工作业绩
#         [0.4, 0.4, 0.2, 0.0],  # 工作态度
#         [0.3, 0.5, 0.2, 0.0],  # 团队协作
#         [0.2, 0.5, 0.3, 0.0]  # 学习能力
#     ])
#     model.set_evaluation_matrix(R)
#
#     # 4. 执行评价并可视化结果
#     result = model.evaluate()
#
#     # 5. 输出详细结果
#     print("\n=== 详细评价结果 ===")
#     print("隶属度分布:")
#     for i, level in enumerate(evaluations):
#         print(f"{level}: {result['membership_degrees'][i]:.4f}")
#
#     print(f"\n最终评价: {result['final_evaluation']}")
#     print(f"综合得分: {result['composite_score']:.1f}分")
#
#     # 6. 结果分析
#     print("\n=== 绩效分析报告 ===")
#     print("1. 该员工整体表现优秀，综合得分较高")
#     print("2. 优势维度：工作业绩突出（优秀率60%）")
#     print("3. 提升空间：学习能力（良好率50%，合格率30%）")
#     print("4. 建议：加强专业技能培训，制定个人发展计划")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class FuzzyEvaluation:
    def __init__(self, factors, evaluations):
        """
        初始化模糊综合评价模型
        :param factors: 评价因素列表
        :param evaluations: 评价等级列表
        """
        self.factors = factors
        self.evaluations = evaluations
        self.weights = None
        self.R_matrix = None

    def set_weights(self, weights):
        """设置权重向量"""
        if abs(sum(weights) - 1.0) > 0.01:
            raise ValueError("权重总和必须为1")
        if len(weights) != len(self.factors):
            raise ValueError("权重数量与因素数量不匹配")
        self.weights = np.array(weights)

    def set_evaluation_matrix(self, R):
        """设置单因素评价矩阵"""
        if R.shape != (len(self.factors), len(self.evaluations)):
            raise ValueError("评价矩阵维度错误")
        self.R_matrix = R

    def evaluate(self, show_result=True):
        """执行模糊综合评价"""
        if self.weights is None or self.R_matrix is None:
            raise ValueError("权重或评价矩阵未设置")

        # 模糊合成运算
        B = np.zeros(len(self.evaluations))
        for j in range(len(self.evaluations)):
            B[j] = sum(self.weights[i] * self.R_matrix[i, j]
                       for i in range(len(self.factors)))

        # 归一化处理
        if np.sum(B) == 0:
            B_normalized = np.ones_like(B) / len(B)  # 防止除零错误
        else:
            B_normalized = B / np.sum(B)

        # 综合评价结果
        eval_index = np.argmax(B_normalized)
        final_evaluation = self.evaluations[eval_index]

        # 计算综合得分（假设评分标准）
        score_standards = [95, 85, 70, 55]  # 优秀/良好/合格/不合格对应分数
        if len(score_standards) != len(B_normalized):
            # 确保评分标准与评价等级数量一致
            score_standards = score_standards[:len(B_normalized)]
        composite_score = sum(B_normalized * np.array(score_standards))

        if show_result:
            self._visualize_results(B_normalized, final_evaluation, composite_score)

        return {
            'membership_degrees': B_normalized,
            'final_evaluation': final_evaluation,
            'composite_score': composite_score
        }

    def _visualize_results(self, B, final_eval, score):
        """可视化评价结果"""
        # 创建图表
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # 子图1：隶属度分布柱状图
        colors = ['#4CAF50', '#8BC34A', '#FFC107', '#F44336']
        ax[0].bar(self.evaluations, B, color=colors[:len(B)])
        ax[0].set_title('各评价等级隶属度分布', fontsize=14)
        ax[0].set_ylabel('隶属度', fontsize=12)
        ax[0].grid(axis='y', linestyle='--', alpha=0.7)

        # 添加数值标签
        for i, v in enumerate(B):
            ax[0].text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)

        # 子图2：雷达图展示各因素评价 - 修复维度问题
        n_factors = len(self.factors)
        angles = np.linspace(0, 2 * np.pi, n_factors, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        # 准备雷达图数据
        radar_data = []
        for i, level in enumerate(self.evaluations):
            # 获取该等级下各因素的评价
            level_data = self.R_matrix[:, i].tolist()
            level_data.append(level_data[0])  # 闭合多边形
            radar_data.append(level_data)

        # 创建雷达图
        ax[1] = plt.subplot(122, polar=True)
        for i, level in enumerate(self.evaluations):
            ax[1].plot(angles, radar_data[i], 'o-', label=level, linewidth=2)

        # 设置角度刻度
        ax[1].set_theta_offset(np.pi / 2)
        ax[1].set_theta_direction(-1)
        ax[1].set_thetagrids(np.degrees(angles[:-1]), self.factors)

        # 设置径向刻度
        ax[1].set_rlabel_position(0)
        ax[1].set_ylim(0, 1)
        ax[1].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

        plt.title('各因素在不同评价等级的分布', fontsize=14, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        # 添加综合结论
        plt.figtext(0.5, 0.05,
                    f"综合评估结果：{final_eval} | 综合得分：{score:.1f}分",
                    ha="center", fontsize=14, bbox={"facecolor": "orange", "alpha": 0.3, "pad": 5})

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig('模糊综合评价结果.png', dpi=300, bbox_inches='tight')
        plt.show()


# ============================= 实例应用 =============================
if __name__ == "__main__":
    print("=== 员工绩效模糊综合评价系统 ===")

    # 1. 初始化评价体系
    factors = ['工作业绩', '工作态度', '团队协作', '学习能力']
    evaluations = ['优秀', '良好', '合格', '不合格']
    model = FuzzyEvaluation(factors, evaluations)

    # 2. 设置权重 (使用AHP方法确定)
    model.set_weights([0.40, 0.25, 0.20, 0.15])

    # 3. 设置评价矩阵 (来自专家评分)
    # 5位专家对员工"张三"的评价统计
    R = np.array([
        [0.6, 0.3, 0.1, 0.0],  # 工作业绩
        [0.4, 0.4, 0.2, 0.0],  # 工作态度
        [0.3, 0.5, 0.2, 0.0],  # 团队协作
        [0.2, 0.5, 0.3, 0.0]  # 学习能力
    ])
    model.set_evaluation_matrix(R)

    # 4. 执行评价并可视化结果
    try:
        result = model.evaluate()

        # 5. 输出详细结果
        print("\n=== 详细评价结果 ===")
        print("隶属度分布:")
        for i, level in enumerate(evaluations):
            print(f"{level}: {result['membership_degrees'][i]:.4f}")

        print(f"\n最终评价: {result['final_evaluation']}")
        print(f"综合得分: {result['composite_score']:.1f}分")

        # 6. 结果分析
        print("\n=== 绩效分析报告 ===")
        print("1. 该员工整体表现优秀，综合得分较高")
        print("2. 优势维度：工作业绩突出（优秀率60%）")
        print("3. 提升空间：学习能力（良好率50%，合格率30%）")
        print("4. 建议：加强专业技能培训，制定个人发展计划")
    except Exception as e:
        print(f"\n!!! 运行时出错: {e}")
        print("建议检查数据维度和参数设置")

