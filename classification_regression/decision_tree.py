"""
ID3&C4.5决策树算法
"""
import math
from collections import Counter, defaultdict

import numpy as np


class node:
    # 这里构建树的节点类，也可用字典来表示树结构
    def __init__(self, fea=-1, res=None, child=None):
        self.fea = fea
        self.res = res
        self.child = child  # 特征的每个值对应一颗子树，特征值为键，相应子树为值


class DecisionTree:
    def __init__(self, epsilon=1e-3, metric='C4.5'):
        self.epsilon = epsilon
        self.tree = None
        self.metric = metric

    def exp_ent(self, y_data):
        # 计算经验熵
        c = Counter(y_data)  # 统计各个类标记的个数
        ent = 0
        N = len(y_data)
        for val in c.values():
            p = val / N
            ent += -p * math.log2(p)
        return ent

    def con_ent(self, fea, X_data, y_data):
        # 计算条件熵并返回，同时返回切分后的各个子数据集
        fea_val_unique = Counter(X_data[:, fea])
        subdata_inds = defaultdict(list)  # 根据特征fea下的值切分数据集
        for ind, sample in enumerate(X_data):
            subdata_inds[sample[fea]].append(ind)  # 挑选某个值对应的所有样本点的索引

        ent = 0
        N = len(y_data)
        for key, val in fea_val_unique.items():
            pi = val / N
            ent += pi * self.exp_ent(y_data[subdata_inds[key]])
        return ent, subdata_inds

    def infoGain(self, fea, X_data, y_data):
        # 计算信息增益
        exp_ent = self.exp_ent(y_data)
        con_ent, subdata_inds = self.con_ent(fea, X_data, y_data)
        return exp_ent - con_ent, subdata_inds

    def infoGainRatio(self, fea, X_data, y_data):
        # 计算信息增益比
        g, subdata_inds = self.infoGain(fea, X_data, y_data)
        N = len(y_data)
        split_info = 0
        for val in subdata_inds.values():
            p = len(val) / N
            split_info -= p * math.log2(p)
        return g / split_info, subdata_inds

    def bestfea(self, fea_list, X_data, y_data):
        # 获取最优切分特征、相应的信息增益（比）以及切分后的子数据集
        score_func = self.infoGainRatio
        if self.metric == 'ID3':
            score_func = self.infoGain
        bestfea = fea_list[0]  # 初始化最优特征
        gmax, bestsubdata_inds = score_func(bestfea, X_data, y_data)  # 初始化最大信息增益及切分后的子数据集
        for fea in fea_list[1:]:
            g, subdata_inds = score_func(fea, X_data, y_data)
            if g > gmax:
                bestfea = fea
                bestsubdata_inds = subdata_inds
                gmax = g
        return gmax, bestfea, bestsubdata_inds

    def buildTree(self, fea_list, X_data, y_data):
        # 递归构建树
        label_unique = np.unique(y_data)
        if label_unique.shape[0] == 1:  # 数据集只有一个类，直接返回该类
            return node(res=label_unique[0])
        if not fea_list:
            return node(res=Counter(y_data).most_common(1)[0][0])
        gmax, bestfea, bestsubdata_inds = self.bestfea(fea_list, X_data, y_data)
        if gmax < self.epsilon:  # 信息增益比小于阈值，返回数据集中出现最多的类
            return node(res=Counter(y_data).most_common(1)[0][0])
        else:
            fea_list.remove(bestfea)
            child = {}
            for key, val in bestsubdata_inds.items():
                child[key] = self.buildTree(fea_list, X_data[val], y_data[val])
            return node(fea=bestfea, child=child)

    def fit(self, X_data, y_data):
        fea_list = list(range(X_data.shape[1]))
        self.tree = self.buildTree(fea_list, X_data, y_data)
        return

    def predict(self, X):
        def helper(X, tree):
            if tree.res is not None:  # 表明到达叶节点
                return tree.res
            else:
                try:
                    sub_tree = tree.child[X[tree.fea]]
                    return helper(X, sub_tree)  # 根据对应特征下的值返回相应的子树
                except:
                    print('input data is out of scope')

        return helper(X, self.tree)


if __name__ == '__main__':
    data = np.array([['青年', '青年', '青年', '青年', '青年', '中年', '中年',
                      '中年', '中年', '中年', '老年', '老年', '老年', '老年', '老年'],
                     ['否', '否', '是', '是', '否', '否', '否', '是', '否',
                      '否', '否', '否', '是', '是', '否'],
                     ['否', '否', '否', '是', '否', '否', '否', '是',
                      '是', '是', '是', '是', '否', '否', '否'],
                     ['一般', '好', '好', '一般', '一般', '一般', '好', '好',
                      '非常好', '非常好', '非常好', '好', '好', '非常好', '一般'],
                     ['否', '否', '是', '是', '否', '否', '否', '是', '是',
                      '是', '是', '是', '是', '是', '否']])
    data = data.T
    X_data = data[:, :-1]
    y_data = data[:, -1]

    import time
    from machine_learning_algorithm.cross_validation import validate
    start = time.clock()

    g = validate(X_data, y_data, ratio=0.2)
    for item in g:
        X_data_train, y_data_train, X_data_test, y_data_test = item
        clf = DecisionTree()
        clf.fit(X_data_train, y_data_train)
        score = 0
        for X, y in zip(X_data_test,y_data_test):
            if clf.predict(X) == y:
                score += 1
        print(score / len(y_data_test))
    print(time.clock() - start)