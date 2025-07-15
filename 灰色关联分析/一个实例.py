import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns#于数据可视化的库

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class GreyRelationalAnalysis:
    def __init__(self, reference_series, compare_series, rho=0.5, method='mean'):
        """
        灰色关联分析
        :param reference_series: 参考序列 (1维数组)
        :param compare_series: 比较序列 (2维数组, 每行一个因素)
        :param rho: 分辨系数 (0~1)
        :param method: 标准化方法 ('mean', 'initial', 'range')
        """
        self.X0 = np.array(reference_series)
        self.Xi = np.array(compare_series)
        self.rho = rho
        self.method = method
        self.n = len(reference_series)  # 数据点数
        self.m = compare_series.shape[0]  # 因素数量
        self.result = None

    def normalize(self):
        """数据标准化"""
        if self.method == 'initial':  # 初值化
            self.X0_norm = self.X0 / self.X0[0]
            self.Xi_norm = self.Xi / self.Xi[:, 0].reshape(-1, 1)

        elif self.method == 'mean':  # 均值化 (默认)
            self.X0_norm = self.X0 / np.mean(self.X0)
            self.Xi_norm = self.Xi / np.mean(self.Xi, axis=1).reshape(-1, 1)

        elif self.method == 'range':  # 区间相对值化
            min_vals = np.min(self.Xi, axis=1)
            max_vals = np.max(self.Xi, axis=1)
            self.X0_norm = (self.X0 - np.min(self.X0)) / (np.max(self.X0) - np.min(self.X0))
            self.Xi_norm = (self.Xi - min_vals.reshape(-1, 1)) / (max_vals.reshape(-1, 1) - min_vals.reshape(-1, 1))

        return self.X0_norm, self.Xi_norm

    def calculate_coefficients(self):
        """计算关联系数"""
        # 计算绝对差
        diff = np.abs(self.Xi_norm - self.X0_norm)

        # 计算两极差
        min_min = np.min(diff)
        max_max = np.max(diff)

        # 计算关联系数矩阵
        coefficients = (min_min + self.rho * max_max) / (diff + self.rho * max_max)

        return coefficients

    def calculate_relations(self):
        """计算关联度并排序"""
        if self.result is None:
            # 标准化数据
            self.normalize()

            # 计算关联系数
            gamma_matrix = self.calculate_coefficients()

            # 计算关联度 (按行求平均)
            relations = np.mean(gamma_matrix, axis=1)

            # 按关联度降序排序
            sorted_idx = np.argsort(-relations)

            self.result = {
                'gamma_matrix': gamma_matrix,
                'relations': relations,
                'sorted_idx': sorted_idx,
                'sorted_relations': relations[sorted_idx]
            }

        return self.result

    def visualize(self, factors_names):
        """可视化分析结果"""
        if self.result is None:
            self.calculate_relations()

        # 创建图表
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 原始数据趋势图
        years = np.arange(2018, 2023)
        ax[0, 0].plot(years, self.X0, 'o-', linewidth=2, markersize=8, label='GDP增长率')
        for i in range(self.m):
            ax[0, 0].plot(years, self.Xi[i], 'o-', label=factors_names[i])
        ax[0, 0].set_title('原始数据趋势图', fontsize=14)
        ax[0, 0].set_xlabel('年份', fontsize=12)
        ax[0, 0].set_ylabel('数值', fontsize=12)
        ax[0, 0].grid(linestyle='--', alpha=0.7)
        ax[0, 0].legend()

        # 2. 标准化数据趋势图
        ax[0, 1].plot(years, self.X0_norm, 'o-', linewidth=2, markersize=8, label='GDP增长率(标准化)')
        for i in range(self.m):
            ax[0, 1].plot(years, self.Xi_norm[i], 'o-', label=f'{factors_names[i]}(标准化)')
        ax[0, 1].set_title('标准化数据趋势图', fontsize=14)
        ax[0, 1].set_xlabel('年份', fontsize=12)
        ax[0, 1].set_ylabel('标准化值', fontsize=12)
        ax[0, 1].grid(linestyle='--', alpha=0.7)
        ax[0, 1].legend()

        # 3. 关联系数热力图
        sns.heatmap(self.result['gamma_matrix'],
                    annot=True, fmt=".3f", cmap="YlGnBu",
                    xticklabels=years,
                    yticklabels=factors_names,
                    ax=ax[1, 0])
        ax[1, 0].set_title('各因素关联系数热力图', fontsize=14)
        ax[1, 0].set_xlabel('年份', fontsize=12)
        ax[1, 0].set_ylabel('影响因素', fontsize=12)

        # 4. 关联度排序图
        sorted_names = [factors_names[i] for i in self.result['sorted_idx']]
        colors = plt.cm.viridis(np.linspace(0, 1, self.m))
        bars = ax[1, 1].barh(sorted_names, self.result['sorted_relations'], color=colors)
        ax[1, 1].set_title('关联度排序', fontsize=14)
        ax[1, 1].set_xlabel('关联度', fontsize=12)
        ax[1, 1].set_xlim(0, 1)

        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax[1, 1].text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                          f'{width:.3f}', ha='left', va='center')

        plt.tight_layout()
        plt.savefig('灰色关联分析结果.png', dpi=300)
        plt.show()

    def print_results(self, factor_names):
        """打印分析结果"""
        if self.result is None:
            self.calculate_relations()

        print("=" * 60)
        print("灰色关联分析报告")
        print("=" * 60)

        # 打印关联系数
        print("\n关联系数矩阵：")
        df_gamma = pd.DataFrame(self.result['gamma_matrix'],
                                index=factor_names,
                                columns=[f"Y{2018 + i}" for i in range(self.n)])
        print(df_gamma)

        # 打印关联度
        print("\n关联度：")
        df_relation = pd.DataFrame({
            '因素': factor_names,
            '关联度': self.result['relations'],
            '排序': [np.where(self.result['sorted_idx'] == i)[0][0] + 1 for i in range(self.m)]
        }).sort_values('关联度', ascending=False)
        print(df_relation)

        # 打印结论
        print("\n分析结论：")
        top_factor = factor_names[self.result['sorted_idx'][0]]
        print(
            f"1. 与GDP增长关联度最高的因素是：{top_factor}（关联度={self.result['relations'][self.result['sorted_idx'][0]]:.4f}）")
        print(f"2. 关联度排序：{' > '.join([factor_names[i] for i in self.result['sorted_idx']])}")
        print("3. 建议：")
        print(f"   - 优先关注{top_factor}对经济增长的拉动作用")
        print("   - 制定针对性政策提升关键因素的发展水平")


# ============================= 实例分析 =============================
if __name__ == "__main__":
    # 1. 准备数据
    # 参考序列 (GDP增长率)
    gdp_growth = np.array([6.8, 6.6, 2.3, 8.1, 5.5])

    # 比较序列 (各影响因素)
    factors = np.array([
        [4500, 4800, 4200, 5200, 5000],  # 固定资产投资
        [3800, 4100, 3500, 4700, 4400],  # 社会消费品零售
        [280, 310, 260, 380, 350],  # 出口总额
        [120, 135, 140, 160, 180]  # 科技创新投入
    ])

    factor_names = ["固定资产投资", "社会消费品零售", "出口总额", "科技创新投入"]

    # 2. 创建并执行灰色关联分析
    gra = GreyRelationalAnalysis(
        reference_series=gdp_growth,
        compare_series=factors,
        rho=0.5,
        method='mean'
    )

    # 3. 计算关联度
    results = gra.calculate_relations()

    # 4. 可视化结果
    gra.visualize(factor_names)

    # 5. 打印详细结果
    gra.print_results(factor_names)


##囫囵吞枣的大致实现了