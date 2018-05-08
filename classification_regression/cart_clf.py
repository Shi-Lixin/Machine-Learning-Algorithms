"""
CART分类树，是一颗二叉树，以某个特征以及该特征对应的一个值为节点，故相对ID3算法，最大的不同就是特征可以使用多次
"""
from collections import Counter, defaultdict

import numpy as np


class node:
    def __init__(self, fea=-1, val=None, res=None, right=None, left=None):
        self.fea = fea  # 特征
        self.val = val  # 特征对应的值
        self.res = res  # 叶节点标记
        self.right = right
        self.left = left


class CART_CLF:
    def __init__(self, epsilon=1e-3, min_sample=1):
        self.epsilon = epsilon
        self.min_sample = min_sample  # 叶节点含有的最少样本数
        self.tree = None

    def getGini(self, y_data):
        # 计算基尼指数
        c = Counter(y_data)
        return 1 - sum([(val / y_data.shape[0]) ** 2 for val in c.values()])

    def getFeaGini(self, set1, set2):
        # 计算某个特征及相应的某个特征值组成的切分节点的基尼指数
        num = set1.shape[0] + set2.shape[0]
        return set1.shape[0] / num * self.getGini(set1) + set2.shape[0] / num * self.getGini(set2)

    def bestSplit(self, splits_set, X_data, y_data):
        # 返回所有切分点的基尼指数，以字典形式存储。键为split，是一个元组，第一个元素为最优切分特征，第二个为该特征对应的最优切分值
        pre_gini = self.getGini(y_data)
        subdata_inds = defaultdict(list)  # 切分点以及相应的样本点的索引
        for split in splits_set:
            for ind, sample in enumerate(X_data):
                if sample[split[0]] == split[1]:
                    subdata_inds[split].append(ind)
        min_gini = 1
        best_split = None
        best_set = None
        for split, data_ind in subdata_inds.items():
            set1 = y_data[data_ind]  # 满足切分点的条件，则为左子树
            set2_inds = list(set(range(y_data.shape[0])) - set(data_ind))
            set2 = y_data[set2_inds]
            if set1.shape[0] < 1 or set2.shape[0] < 1:
                continue
            now_gini = self.getFeaGini(set1, set2)
            if now_gini < min_gini:
                min_gini = now_gini
                best_split = split
                best_set = (data_ind, set2_inds)
        if abs(pre_gini - min_gini) < self.epsilon:  # 若切分后基尼指数下降未超过阈值则停止切分
            best_split = None
        return best_split, best_set, min_gini

    def buildTree(self, splits_set, X_data, y_data):
        if y_data.shape[0] < self.min_sample:  # 数据集小于阈值直接设为叶节点
            return node(res=Counter(y_data).most_common(1)[0][0])
        best_split, best_set, min_gini = self.bestSplit(splits_set, X_data, y_data)
        if best_split is None:  # 基尼指数下降小于阈值，则终止切分，设为叶节点
            return node(res=Counter(y_data).most_common(1)[0][0])
        else:
            splits_set.remove(best_split)
            left = self.buildTree(splits_set, X_data[best_set[0]], y_data[best_set[0]])
            right = self.buildTree(splits_set, X_data[best_set[1]], y_data[best_set[1]])
            return node(fea=best_split[0], val=best_split[1], right=right, left=left)

    def fit(self, X_data, y_data):
        # 训练模型，CART分类树与ID3最大的不同是，CART建立的是二叉树，每个节点是特征及其对应的某个值组成的元组
        # 特征可以多次使用
        splits_set = []
        for fea in range(X_data.shape[1]):
            unique_vals = np.unique(X_data[:, fea])
            if unique_vals.shape[0] < 2:
                continue
            elif unique_vals.shape[0] == 2:  # 若特征取值只有2个，则只有一个切分点，非此即彼
                splits_set.append((fea, unique_vals[0]))
            else:
                for val in unique_vals:
                    splits_set.append((fea, val))
        self.tree = self.buildTree(splits_set, X_data, y_data)
        return

    def predict(self, x):
        def helper(x, tree):
            if tree.res is not None:  # 表明到达叶节点
                return tree.res
            else:
                if x[tree.fea] == tree.val:  # "是" 返回左子树
                    branch = tree.left
                else:
                    branch = tree.right
                return helper(x, branch)

        return helper(x, self.tree)

    def disp_tree(self):
        # 打印树
        self.disp_helper(self.tree)
        return

    def disp_helper(self, current_node):
        # 前序遍历
        print(current_node.fea, current_node.val, current_node.res)
        if current_node.res is not None:
            return
        self.disp_helper(current_node.left)
        self.disp_helper(current_node.right)
        return


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    X_data = load_iris().data
    y_data = load_iris().target

    from machine_learning_algorithm.cross_validation import validate

    g = validate(X_data, y_data, ratio=0.2)
    for item in g:
        X_data_train, y_data_train, X_data_test, y_data_test = item
        clf = CART_CLF()
        clf.fit(X_data_train, y_data_train)
        score = 0
        for X, y in zip(X_data_test,y_data_test):
            if clf.predict(X) == y:
                score += 1
        print(score / len(y_data_test))
