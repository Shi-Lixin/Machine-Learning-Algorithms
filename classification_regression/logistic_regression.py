"""
逻辑斯谛回归
"""

import numpy as np


class LR:
    def __init__(self, alpha=0.01, maxstep=1000):
        self.w = None
        self.maxstep = maxstep
        self.alpha = alpha

    def sig(self, z):
        # Logistic函数, 正类的概率
        return 1.0 / (1 + np.exp(-z))

    def bgd(self, X_data, y_data):  # 损失函数采用对数损失函数，其数学形式与似然函数一致
        # 批量梯度下降法
        b = np.ones((X_data.shape[0], 1))
        X = np.hstack((X_data, b))  # 考虑阈值，堆输入向量进行扩充
        w = np.ones(X.shape[1])  # 初始化各特征的权重
        i = 0
        while i <= self.maxstep:
            i += 1
            err = y_data - self.sig(w @ X.T)
            w += self.alpha * err @ X  # 注意，其表达式与平方误差损失函数的非常相似，但这是由对数损失函数推导而来的
        self.w = w
        return

    def fit(self, X_data, y_data):
        self.bgd(X_data, y_data)
        return

    def predict(self, x):
        x = np.append(x, 1)
        PT = self.sig(self.w @ x.T)
        if PT > 1 - PT:
            return 1
        else:
            return 0


if __name__ == '__main__':
    from sklearn import datasets

    data = datasets.load_digits(n_class=2)
    X_data = data['data']
    y_data = data['target']
    from machine_learning_algorithm.cross_validation import validate
    g = validate(X_data, y_data, ratio=0.2)
    for item in g:
        X_train, y_train, X_test, y_test = item
        clf = LR()
        clf.fit(X_train, y_train)
        score = 0
        for x, y in zip(X_test, y_test):
            if clf.predict(x)==y:
                score += 1
        print(score/len(y_test))