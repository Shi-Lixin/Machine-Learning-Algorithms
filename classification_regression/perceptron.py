"""
感知机(perceptron):原始形式以及对偶形式
"""
import numpy as np


class Perceptron:
    def __init__(self, eta=1):
        self.eta = eta  # 学习率
        self.w = None  # 权值
        self.b = None  # 阈值

    def fit(self, X_data, y_data):
        self.w = np.zeros(X_data.shape[1])  # 初始化
        self.b = 0
        change = True
        while change:  # w, b 不发生改变则结束训练
            for X, y in zip(X_data, y_data):  # 依次输入每个数据点进行训练
                change = False
                while y * (self.w @ X + self.b) <= 0:
                    self.w += self.eta * X * y
                    self.b += self.eta * y
                    change = True
        return

    def predict(self, X):
        return np.sign(self.w @ X + self.b)


class Perceptron_dual:
    # 对偶形式的感知机
    def __init__(self, eta=1):
        self.eta = eta
        self.alpha = None  # alpha相当于样本的权值，当eta为1时就是每个样本参与训练的次数
        self.b = None

        self.N = None
        self.gram = None

    def init_param(self, X_data):
        self.N = X_data.shape[0]
        self.alpha = np.zeros(self.N)
        self.b = 0
        self.gram = self.getGram(X_data)

    def getGram(self, X_data):
        # 计算Gram矩阵
        gram = np.diag(np.linalg.norm(X_data, axis=1) ** 2)

        for i in range(self.N):
            for j in range(i + 1, self.N):
                gram[i, j] = X_data[i] @ X_data[j]
                gram[j, i] = gram[i, j]

        return gram

    def sum_dual(self, y_data, i):
        s = 0
        for j in range(self.N):
            s += self.alpha[j] * y_data[j] * self.gram[j][i]
        return y_data[i] * (s + self.b)

    def fit(self, X_data, y_data):
        self.init_param(X_data)
        changed = True
        while changed:
            changed = False
            for i in range(self.N):  # 依次输入每个数据点进行训练
                while self.sum_dual(y_data, i) <= 0:
                    self.alpha[i] += self.eta
                    self.b += self.eta * y_data[i]
                    changed = True
        return


if __name__ == '__main__':
    X_data = np.array([[3, 3], [4, 3], [1, 1]])
    y_data = np.array([1, 1, -1])
    p = Perceptron()
    p.fit(X_data, y_data)
    print(p.w, p.b)
