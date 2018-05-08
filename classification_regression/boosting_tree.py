"""
提升树：基于二叉回归树的提升算法
程序暂考虑输入为一维的情况
"""
from collections import defaultdict
import numpy as np


class BoostingTree:
    def __init__(self, epsilon=1e-2):
        self.epsilon = epsilon
        self.cand_splits = []  # 候选切分点
        self.split_index = defaultdict(tuple)  # 由于要多次切分数据集，故预先存储，切分后数据点的索引
        self.split_list = []  # 最终各个基本回归树的切分点
        self.c1_list = []  # 切分点左区域取值
        self.c2_list = []  # 切分点右区域取值
        self.N = None
        self.n_split = None

    def init_param(self, X_data):
        # 初始化参数
        self.N = X_data.shape[0]
        for i in range(1, self.N):
            self.cand_splits.append((X_data[i][0] + X_data[i - 1][0]) / 2)
        self.n_split = len(self.cand_splits)
        for split in self.cand_splits:
            left_index = np.where(X_data[:, 0]<= split)[0]
            right_index = list(set(range(self.N))-set(left_index))
            self.split_index[split] = (left_index, right_index)
        return

    def _cal_err(self, split, y_res):
        # 计算每个切分点的误差
        inds = self.split_index[split]
        left = y_res[inds[0]]
        right = y_res[inds[1]]

        c1 = np.sum(left) / len(left)
        c2 = np.sum(right) / len(right)
        y_res_left = left - c1
        y_res_right = right - c2
        res = np.hstack([y_res_left, y_res_right])
        res_square = np.apply_along_axis(lambda x: x ** 2, 0, res).sum()
        return res_square, c1, c2

    def best_split(self,y_res):
        # 获取最佳切分点，并返回对应的残差
        best_split = self.cand_splits[0]
        min_res_square, best_c1, best_c2 = self._cal_err(best_split, y_res)

        for i in range(1, self.n_split):
            res_square, c1, c2 = self._cal_err(self.cand_splits[i], y_res)
            if res_square < min_res_square:
                best_split = self.cand_splits[i]
                min_res_square = res_square
                best_c1 = c1
                best_c2 = c2

        self.split_list.append(best_split)
        self.c1_list.append(best_c1)
        self.c2_list.append(best_c2)
        return

    def _fx(self, X):
        # 基于当前组合树，预测X的输出值
        s = 0
        for split, c1, c2 in zip(self.split_list, self.c1_list, self.c2_list):
            if X < split:
                s += c1
            else:
                s += c2
        return s

    def update_y(self, X_data, y_data):
        # 每添加一颗回归树，就要更新y,即基于当前组合回归树的预测残差
        y_res = []
        for X, y in zip(X_data, y_data):
            y_res.append(y - self._fx(X[0]))
        y_res = np.array(y_res)
        res_square = np.apply_along_axis(lambda x: x ** 2, 0, y_res).sum()
        return y_res, res_square

    def fit(self, X_data, y_data):
        self.init_param(X_data)
        y_res = y_data
        while True:
            self.best_split(y_res)
            y_res, res_square = self.update_y(X_data, y_data)
            if res_square < self.epsilon:
                break
        return

    def predict(self, X):
        return self._fx(X)


if __name__ == '__main__':
    # data = np.array(
    #     [[1, 5.56], [2, 5.70], [3, 5.91], [4, 6.40], [5, 6.80], [6, 7.05], [7, 8.90], [8, 8.70], [9, 9.00], [10, 9.05]])
    # X_data = data[:, :-1]
    # y_data = data[:, -1]
    # BT = BoostingTree(epsilon=0.18)
    # BT.fit(X_data, y_data)
    # print(BT.split_list, BT.c1_list, BT.c2_list)
    X_data_raw = np.linspace(-5, 5, 100)
    X_data = np.transpose([X_data_raw])
    y_data = np.sin(X_data_raw)
    BT = BoostingTree(epsilon=0.1)
    BT.fit(X_data, y_data)
    y_pred = [BT.predict(X) for X in X_data]

    import matplotlib.pyplot as plt

    p1 = plt.scatter(X_data_raw, y_data, color='r')
    p2 = plt.scatter(X_data_raw, y_pred, color='b')
    plt.legend([p1, p2], ['real', 'pred'])
    plt.show()
