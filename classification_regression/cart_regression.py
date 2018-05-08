"""
CART+最小二乘法构建CART回归树
"""

import numpy as np


class node:
    def __init__(self, fea=-1, val=None, res=None, right=None, left=None):
        self.fea = fea
        self.val = val
        self.res = res
        self.right = right
        self.left = left


class CART_REG:
    def __init__(self, epsilon=0.1, min_sample=10):
        self.epsilon = epsilon
        self.min_sample = min_sample
        self.tree = None

    def err(self, y_data):
        # 子数据集的输出变量y与均值的差的平方和
        return y_data.var() * y_data.shape[0]

    def leaf(self, y_data):
        # 叶节点取值，为子数据集输出y的均值
        return y_data.mean()

    def split(self, fea, val, X_data):
        # 根据某个特征，以及特征下的某个取值，将数据集进行切分
        set1_inds = np.where(X_data[:, fea] <= val)[0]
        set2_inds = list(set(range(X_data.shape[0]))-set(set1_inds))
        return set1_inds, set2_inds

    def getBestSplit(self, X_data, y_data):
        # 求最优切分点
        best_err = self.err(y_data)
        best_split = None
        subsets_inds = None
        for fea in range(X_data.shape[1]):
            for val in X_data[:, fea]:
                set1_inds, set2_inds = self.split(fea, val, X_data)
                if len(set1_inds) < 2 or len(set2_inds) < 2:  # 若切分后某个子集大小不足2，则不切分
                    continue
                now_err = self.err(y_data[set1_inds]) + self.err(y_data[set2_inds])
                if now_err < best_err:
                    best_err = now_err
                    best_split = (fea, val)
                    subsets_inds = (set1_inds, set2_inds)
        return best_err, best_split, subsets_inds

    def buildTree(self, X_data, y_data):
        # 递归构建二叉树
        if y_data.shape[0] < self.min_sample:
            return node(res=self.leaf(y_data))
        best_err, best_split, subsets_inds = self.getBestSplit(X_data, y_data)
        if subsets_inds is None:
            return node(res=self.leaf(y_data))
        if best_err < self.epsilon:
            return node(res=self.leaf(y_data))
        else:
            left = self.buildTree(X_data[subsets_inds[0]], y_data[subsets_inds[0]])
            right = self.buildTree(X_data[subsets_inds[1]], y_data[subsets_inds[1]])
            return node(fea=best_split[0], val=best_split[1], right=right, left=left)

    def fit(self, X_data, y_data):
        self.tree = self.buildTree(X_data, y_data)
        return

    def predict(self, x):
        # 对输入变量进行预测
        def helper(x, tree):
            if tree.res is not None:
                return tree.res
            else:
                if x[tree.fea] <= tree.val:
                    branch = tree.left
                else:
                    branch = tree.right
                return helper(x, branch)

        return helper(x, self.tree)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X_data_raw = np.linspace(-3, 3, 50)
    np.random.shuffle(X_data_raw)
    y_data = np.sin(X_data_raw)
    X_data = np.transpose([X_data_raw])
    y_data = y_data + 0.1 * np.random.randn(y_data.shape[0])
    clf = CART_REG(epsilon=1e-4, min_sample=1)
    clf.fit(X_data, y_data)
    res = []
    for i in range(X_data.shape[0]):
        res.append(clf.predict(X_data[i]))
    p1 = plt.scatter(X_data_raw, y_data)
    p2 = plt.scatter(X_data_raw, res, marker='*')
    plt.legend([p1,p2],['real','pred'],loc='upper left')
    plt.show()
