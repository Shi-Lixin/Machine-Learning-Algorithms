"""
SoftMax回归，逻辑斯蒂回归的多分类推广。所以，本质还是一种分类算法
"""
import numpy as np


class SoftMax:
    def __init__(self, maxstep=10000, C=1e-4, alpha=0.4):
        self.maxstep = maxstep
        self.C = C  # 权值衰减项系数lambda, 类似于惩罚系数
        self.alpha = alpha  # 学习率

        self.w = None  # 权值

        self.L = None  # 类的数量
        self.D = None  # 输入数据维度
        self.N = None  # 样本总量

    def init_param(self, X_data, y_data):
        # 初始化，暂定输入数据全部为数值形式
        b = np.ones((X_data.shape[0], 1))
        X_data = np.hstack((X_data, b))  # 附加偏置项
        self.L = len(np.unique(y_data))
        self.D = X_data.shape[1]
        self.N = X_data.shape[0]
        self.w = np.ones((self.L, self.D))  # l*d, 针对每个类，都有一组权值参数w
        return X_data

    def bgd(self, X_data, y_data):
        # 梯度下降训练
        step = 0
        while step < self.maxstep:
            step += 1
            prob = np.exp(X_data @ self.w.T)  # n*l, 行向量存储该样本属于每个类的概率
            nf = np.transpose([prob.sum(axis=1)])  # n*1
            nf = np.repeat(nf, self.L, axis=1)  # n*l
            prob = -prob / nf  # 归一化， 此处条件符号仅方便后续计算梯度
            for i in range(self.N):
                prob[i, int(y_data[i])] += 1
            grad = -1.0 / self.N * prob.T @ X_data + self.C * self.w  # 梯度， 第二项为衰减项
            self.w -= self.alpha * grad
        return

    def fit(self, X_data, y_data):
        X_data = self.init_param(X_data, y_data)
        self.bgd(X_data, y_data)
        return

    def predict(self, X):
        b = np.ones((X.shape[0], 1))
        X = np.hstack((X, b))  # 附加偏置项
        prob = np.exp(X @ self.w.T)
        return np.argmax(prob, axis=1)


if __name__ == '__main__':
    from sklearn.datasets import load_digits

    data = load_digits()
    X_data = data['data']
    y_data = data['target']

    from machine_learning_algorithm.cross_validation import validate

    g = validate(X_data, y_data, ratio=0.2)
    for item in g:
        X_train, y_train, X_test, y_test = item
        clf = SoftMax(maxstep=10000, alpha=0.1, C=1e-4)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = 0
        for y, y_pred in zip(y_test, y_pred):
            score += 1 if y == y_pred else 0
        print(score / len(y_test))
