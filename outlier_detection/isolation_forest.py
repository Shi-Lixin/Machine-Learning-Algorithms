# 实现IsolationForest高维数据的异常值检测算法
import numpy as np
import math
from collections import Counter


class Node:
    def __init__(self, val=None, right=None, left=None):
        self.val = val  # 存储样本索引,仅叶节点
        self.right = right
        self.left = left


class RandomTree:
    def __init__(self):
        self.tree = None
        self.n_feas = None

    def get_split(self, data, inds):
        # 随机构建切分点
        f = np.random.choice(self.n_feas)  # 随机选择一个特征
        up = max(data[inds, f])
        down = min(data[inds, f])
        v = (up - down) * np.random.sample() + down  # 在该特征的最大与最小值间随机选择一个数
        return f, v

    def split(self, data, inds):
        # 切分数据集
        f, v = self.get_split(data, inds)
        left_ind = []
        right_ind = []
        for i in inds:
            if data[i, f] <= v:
                left_ind.append(i)
            else:
                right_ind.append(i)
        return left_ind, right_ind

    def buildTree(self, data, inds):
        if len(inds) < 3:  # 叶节点
            return Node(val=inds)
        left_ind, right_ind = self.split(data, inds)
        left = self.buildTree(data, left_ind)
        right = self.buildTree(data, right_ind)
        return Node(left=left, right=right)

    def fit(self, data):
        self.n_feas = data.shape[1]
        inds = np.arange(data.shape[0])
        self.tree = self.buildTree(data, inds)
        return

    def traverse(self):
        # 遍历树，统计每个样本的路径长
        path_len = Counter()
        i = -1

        def helper(currentNode):
            nonlocal i
            i += 1
            if currentNode.val is not None:
                for ind in currentNode.val:
                    path_len[ind] = i
                return
            for child in [currentNode.left, currentNode.right]:
                helper(child)
                i -= 1
            return

        helper(self.tree)
        return path_len


class IsolationForest:
    def __init__(self, n_tree, epsilon):
        self.n_tree = n_tree
        self.epsilon = epsilon  # 异常点比例
        self.scores = Counter()

    def fit_predict(self, data):
        for _ in range(self.n_tree):
            RT = RandomTree()
            RT.fit(data)
            path_len = RT.traverse()
            self.scores = self.scores + path_len

        n_sample = data.shape[0]
        phi = 2 * math.log(n_sample - 1) - 2 * (n_sample - 1) / n_sample
        for key, val in self.scores.items():
            self.scores[key] = 2 ** -(val / self.n_tree / phi)  # 归一化
        q = np.quantile(list(self.scores.values()), 1 - self.epsilon)
        outliers = [key for key, val in self.scores.items() if val > q]
        return outliers


if __name__ == '__main__':
    np.random.seed(42)
    X_inliers = 0.3 * np.random.randn(100, 2)
    X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    data = np.r_[X_inliers, X_outliers]

    IF = IsolationForest(100, 0.1)
    out_ind = IF.fit_predict(data)
    outliers = data[out_ind]

    import matplotlib.pyplot as plt

    plt.scatter(data[:, 0], data[:, 1], color='b')
    plt.scatter(outliers[:, 0], outliers[:, 1], color='r')
    plt.show()
