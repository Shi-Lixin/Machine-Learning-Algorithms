"""
支持向量机
学习方法:不等式约束的最优化问题,求解凸二次规划的最优化算法, 本程序采用序列最小最优化算法(SOM)
"""
import numpy as np
import math


class SVM:
    def __init__(self, epsilon=1e-5, maxstep=500, C=1.0, kernel_option=True, gamma=None):
        self.epsilon = epsilon
        self.maxstep = maxstep
        self.C = C
        self.kernel_option = kernel_option  # 是否选择核函数
        self.gamma = gamma  # 高斯核参数

        self.kernel_arr = None  # n*n 存储核内积
        self.Q = None  # n*n yi*yj*K(i,j)
        self.grad = None  # 1*n 存储每个alpha的梯度，相对于对偶问题的最优化函数而言
        self.X = None  # 训练数据集
        self.y = None  # 类标记值，是计算w,b的参数，故存入模型中
        self.alpha_arr = None  # 1*n 存储拉格朗日乘子, 每个样本对应一个拉格朗日乘子
        self.b = 0  # 阈值b, 初始化为0

        self.N = None

    def init_param(self, X_data, y_data):
        # 初始化参数, 包括核内积矩阵、alpha和预测误差
        self.N = X_data.shape[0]
        self.X = X_data
        self.y = y_data
        if self.gamma is None:
            self.gamma = 1.0 / X_data.shape[1]
        self.cal_kernel(X_data)
        self.alpha_arr = np.zeros(self.N)
        self.grad = - np.ones(self.N)
        _y = np.array([y_data])  # 1*n
        self.Q = _y.T @ _y * self.kernel_arr
        return

    def _gaussian_dot(self, x1, x2):
        # 计算两个样本之间的高斯内积
        return math.exp(-self.gamma * np.square(x1 - x2).sum())

    def cal_kernel(self, X_data):
        # 计算核内积矩阵
        if self.kernel_option:
            self.kernel_arr = np.ones((self.N, self.N))
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    self.kernel_arr[i, j] = self._gaussian_dot(X_data[i], X_data[j])
                    self.kernel_arr[j, i] = self.kernel_arr[i, j]
        else:
            self.kernel_arr = X_data @ X_data.T  # 不使用高斯核，线性分类器
        return

    def get_working_set(self, y_data):
        # 挑选两个变量alpha, 返回索引
        ind1 = -1
        ind2 = -1
        max_grad = - float('inf')
        min_grad = float('inf')
        # 挑选第一个alpha, 正类
        for i in range(self.N):
            if (y_data[i] == 1 and self.alpha_arr[i] < self.C) or (y_data[i] == -1 and self.alpha_arr[i] > 0):
                tmp = -y_data[i] * self.grad[i]
                if tmp >= max_grad:
                    ind1 = i
                    max_grad = tmp
        # 挑选第二个alpha, 负类
        ab_obj = float('inf')
        for i in range(self.N):
            if (y_data[i] == 1 and self.alpha_arr[i] > 0) or (y_data[i] == -1 and self.alpha_arr[i] < self.C):
                tmp = y_data[i] * self.grad[i]
                b = max_grad + tmp
                if -tmp < min_grad:
                    min_grad = -tmp
                if b > 0:
                    a = self.Q[ind1][ind1] + self.Q[i][i] - 2 * y_data[ind1] * y_data[i] * \
                        self.Q[ind1][i]
                    if a <= 0:
                        a = 1e-12
                    if - b ** 2 / a < ab_obj:
                        ind2 = i
                        ab_obj = - b ** 2 / a
        if max_grad - min_grad >= self.epsilon:  # 收敛条件
            return ind1, ind2
        return -1, -1

    def update(self, ind1, ind2):
        # 更新挑选出的两个样本的alpha、对应的预测值及误差和阈值b
        old_alpha1 = self.alpha_arr[ind1]
        old_alpha2 = self.alpha_arr[ind2]
        y1 = self.y[ind1]
        y2 = self.y[ind2]
        a = self.Q[ind1][ind1] + self.Q[ind2][ind2] - 2 * y1 * y2 * self.Q[ind1][ind2]
        if a <= 0:
            a = 1e-12
        b = -y1 * self.grad[ind1] + y2 * self.grad[ind2]
        new_alpha1 = old_alpha1 + y1 * b / a
        # 剪辑
        s = y1 * old_alpha1 + y2 * old_alpha2
        if new_alpha1 > self.C:
            new_alpha1 = self.C
        if new_alpha1 < 0:
            new_alpha1 = 0
        new_alpha2 = y2 * (s - y1 * new_alpha1)
        if new_alpha2 > self.C:
            new_alpha2 = self.C
        if new_alpha2 < 0:
            new_alpha2 = 0
        new_alpha1 = y1 * (s - y2 * new_alpha2)
        self.alpha_arr[ind1] = new_alpha1
        self.alpha_arr[ind2] = new_alpha2
        # 更新梯度
        delta1 = new_alpha1 - old_alpha1
        delta2 = new_alpha2 - old_alpha2
        for i in range(self.N):
            self.grad[i] += self.Q[i][ind1] * delta1 + self.Q[i][ind2] * delta2
        return

    def fit(self, X_data, y_data):
        # 训练主函数
        self.init_param(X_data, y_data)
        step = 0
        while step < self.maxstep:
            step += 1
            ind1, ind2 = self.get_working_set(y_data)
            if ind2 == -1:
                break
            self.update(ind1, ind2)
        # 计算阈值b
        alpha0_inds = set(np.where(self.grad == 0)[0])
        alphaC_inds = set(np.where(self.grad == self.C)[0])
        alpha_inds = set(range(self.N)) - alphaC_inds - alpha0_inds

        label_inds1 = set(np.where(y_data == 1)[0])
        r1_inds = list(label_inds1 & alpha_inds)
        if r1_inds:
            r1 = self.grad[r1_inds].sum()
        else:
            min_r1 = self.grad[list(alpha0_inds & label_inds1)].min()
            max_r1 = self.grad[list(alphaC_inds & label_inds1)].max()
            r1 = (min_r1 + max_r1) / 2

        label_inds2 = set(np.where(y_data == -1)[0])
        r2_inds = list(label_inds1 & alpha_inds)
        if r2_inds:
            r2 = self.grad[r2_inds].sum()
        else:
            min_r2 = self.grad[list(alpha0_inds & label_inds2)].min()
            max_r2 = self.grad[list(alphaC_inds & label_inds2)].max()
            r2 = (min_r2 + max_r2) / 2
        self.b = (r2 - r1) / 2
        return

    def predict(self, x):
        # 预测x的类别
        if self.kernel_option:
            kernel = np.array([self._gaussian_dot(x, sample) for sample in self.X])
            g = np.sum(self.y * self.alpha_arr * kernel)
        else:
            g = np.sum(self.alpha_arr * self.y * (np.array([x]) @ self.X.T)[0])
        return np.sign(g + self.b)


if __name__ == "__main__":
    from sklearn.datasets import load_digits

    data = load_digits(n_class=2)
    X_data = data['data']
    y_data = data['target']
    inds = np.where(y_data == 0)[0]
    y_data[inds] = -1

    from machine_learning_algorithm.cross_validation import validate

    g = validate(X_data, y_data)
    for item in g:
        X_train, y_train, X_test, y_test = item
        S = SVM(kernel_option=False, maxstep=1000, epsilon=1e-3, C=1.0)
        S.fit(X_train, y_train)
        score = 0
        for X, y in zip(X_test, y_test):
            if S.predict(X) == y:
                score += 1
        print(score / len(y_test))
