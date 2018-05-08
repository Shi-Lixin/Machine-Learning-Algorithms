"""
随机森林算法，组合算法bagging(装袋)的一种
"""
from collections import defaultdict
import numpy as np
import math
from cart_clf import CART_CLF


class RandomForest:
    def __init__(self, n_tree=6, n_fea=None, ri_rc=True, L=None, epsilon=1e-3, min_sample=1):
        self.n_tree = n_tree
        self.n_fea = n_fea  # 每棵树中特征的数量
        self.ri_rc = ri_rc  # 判定特征的选择选用RI还是RC, 特征比较少时使用RC
        self.L = L # 选择RC时，进行线性组合的特征个数
        self.tree_list = []  # 随机森林中子树的list

        self.epsilon = epsilon
        self.min_sample = min_sample  # 叶节点含有的最少样本数

        self.D = None  # 输入数据维度
        self.N = None

    def init_param(self, X_data):
        # 初始化参数
        self.D = X_data.shape[1]
        self.N = X_data.shape[0]
        if self.n_fea is None:
            self.n_fea = int(math.log2(self.D) + 1)  # 默认选择特征的个数
        return

    def extract_fea(self):
        # 从原数据中抽取特征(RI)或线性组合构建新特征(RC)
        if self.ri_rc:
            if self.n_fea > self.D:
                raise ValueError('the number of features should be lower than dimention of data while RI is chosen')
            fea_arr = np.random.choice(self.D, self.n_fea, replace=False)
        else:
            fea_arr = np.zeros((self.n_fea, self.D))
            for i in range(self.n_fea):
                out_fea = np.random.choice(self.D, self.L, replace=False)
                coeff = np.random.uniform(-1, 1, self.D)  # [-1,1]上的均匀分布来产生每个特征前的系数
                coeff[out_fea] = 0
                fea_arr[i] = coeff
        return fea_arr

    def extract_data(self, X_data, y_data):
        # 从原数据中有放回的抽取样本，构成每个决策树的自助样本集
        fea_arr = self.extract_fea()  # col_index or coeffs
        inds = np.unique(np.random.choice(self.N, self.N))  # row_index, 有放回抽取样本
        sub_X = X_data[inds]
        sub_y = y_data[inds]
        if self.ri_rc:
            sub_X = sub_X[:, fea_arr]
        else:
            sub_X = sub_X @ fea_arr.T
        return sub_X, sub_y, fea_arr

    def fit(self, X_data, y_data):
        # 训练主函数
        self.init_param(X_data)
        for i in range(self.n_tree):
            sub_X, sub_y, fea_arr = self.extract_data(X_data, y_data)
            subtree = CART_CLF(epsilon=self.epsilon, min_sample=self.min_sample)
            subtree.fit(sub_X, sub_y)
            self.tree_list.append((subtree, fea_arr))  # 保存训练后的树及其选用的特征，以便后续预测时使用
        return

    def predict(self, X):
        # 预测，多数表决
        res = defaultdict(int)  # 存储每个类得到的票数
        for item in self.tree_list:
            subtree, fea_arr = item
            if self.ri_rc:
                X_modify = X[fea_arr]
            else:
                X_modify = (np.array([X]) @ fea_arr.T)[0]
            label = subtree.predict(X_modify)
            res[label] += 1
        return max(res, key=res.get)


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    data = load_iris()
    X_data = data['data']
    y_data = data['target']
    from machine_learning_algorithm.cross_validation import validate

    g = validate(X_data, y_data, ratio=0.2)
    for item in g:
        X_train, y_train, X_test, y_test = item
        RF = RandomForest(n_tree=50, n_fea=2, ri_rc=True)
        RF.fit(X_train, y_train)
        score = 0
        for X, y in zip(X_test, y_test):
            if RF.predict(X) == y:
                score += 1
        print(score / len(y_test))
