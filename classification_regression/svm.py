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
        self.X = None  # 训练数据集
        self.y = None  # 类标记值，是计算w,b的参数，故存入模型中
        self.alpha_arr = None  # 1*n 存储拉格朗日乘子, 每个样本对应一个拉格朗日乘子
        self.b = 0  # 阈值b, 初始化为0
        self.err_arr = None  # 1*n  记录每个样本的预测误差

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
        self.err_arr = - self.y  # 将拉格朗日乘子全部初始化为0，则相应的预测值初始化为0，预测误差就是-y_data
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

    def select_second_alpha(self, ind1):
        # 挑选第二个变量alpha, 返回索引
        E1 = self.err_arr[ind1]
        ind2 = None
        max_diff = 0  # 初始化最大的|E1-E2|
        candidate_alpha_inds = np.nonzero(self.err_arr)[0]  # 存在预测误差的样本作为候选样本
        if len(candidate_alpha_inds) > 1:
            for i in candidate_alpha_inds:
                if i == ind1:
                    continue
                tmp = abs(self.err_arr[i] - E1)
                if tmp > max_diff:
                    max_diff = tmp
                    ind2 = i
        if ind2 is None:  # 随机选择一个不与ind1相等的样本索引
            ind2 = ind1
            while ind2 == ind1:
                ind2 = np.random.choice(self.N)
        return ind2

    def update(self, ind1, ind2):
        # 更新挑选出的两个样本的alpha、对应的预测值及误差和阈值b
        old_alpha1 = self.alpha_arr[ind1]
        old_alpha2 = self.alpha_arr[ind2]
        y1 = self.y[ind1]
        y2 = self.y[ind2]
        if y1 == y2:
            L = max(0.0, old_alpha2 + old_alpha1 - self.C)
            H = min(self.C, old_alpha2 + old_alpha1)
        else:
            L = max(0.0, old_alpha2 - old_alpha1)
            H = min(self.C, self.C + old_alpha2 - old_alpha1)
        if L == H:
            return 0
        E1 = self.err_arr[ind1]
        E2 = self.err_arr[ind2]
        K11 = self.kernel_arr[ind1, ind1]
        K12 = self.kernel_arr[ind1, ind2]
        K22 = self.kernel_arr[ind2, ind2]
        # 更新alpha2
        eta = K11 + K22 - 2 * K12
        if eta <= 0:
            return 0
        new_unc_alpha2 = old_alpha2 + y2 * (E1 - E2) / eta  # 未经剪辑的alpha2
        if new_unc_alpha2 > H:
            new_alpha2 = H
        elif new_unc_alpha2 < L:
            new_alpha2 = L
        else:
            new_alpha2 = new_unc_alpha2
        # 更新alpha1
        if abs(old_alpha2 - new_alpha2) < self.epsilon * (
                old_alpha2 + new_alpha2 + self.epsilon):  # 若alpha2更新变化很小，则忽略本次更新
            return 0
        new_alpha1 = old_alpha1 + y1 * y2 * (old_alpha2 - new_alpha2)
        self.alpha_arr[ind1] = new_alpha1
        self.alpha_arr[ind2] = new_alpha2
        # 更新阈值b
        new_b1 = -E1 - y1 * K11 * (new_alpha1 - old_alpha1) - y2 * K12 * (new_alpha2 - old_alpha2) + self.b
        new_b2 = -E2 - y1 * K12 * (new_alpha1 - old_alpha1) - y2 * K22 * (new_alpha2 - old_alpha2) + self.b
        if 0 < new_alpha1 < self.C:
            self.b = new_b1
        elif 0 < new_alpha2 < self.C:
            self.b = new_b2
        else:
            self.b = (new_b1 + new_b2) / 2
        # 更新对应的预测误差
        self.err_arr[ind1] = np.sum(self.y * self.alpha_arr * self.kernel_arr[ind1, :]) + self.b - y1
        self.err_arr[ind2] = np.sum(self.y * self.alpha_arr * self.kernel_arr[ind2, :]) + self.b - y2
        return 1

    def satisfy_kkt(self, y, err, alpha):
        # 在精度范围内判断是否满足KTT条件
        r = y * err
        # r<=0,则y(g-y)<=0,yg<1, alpha=C则符合；r>0,则yg>1, alpha=0则符合
        if (r < -self.epsilon and alpha < self.C) or (r > self.epsilon and alpha > 0):
            return False
        return True

    def fit(self, X_data, y_data):
        # 训练主函数
        self.init_param(X_data, y_data)
        # 启发式搜索第一个alpha时，当间隔边界上的支持向量全都满足KKT条件时，就搜索整个数据集。
        # 整个训练过程需要在边界支持向量与所有样本集之间进行切换搜索，以防止无法收敛
        entire_set = True
        step = 0
        change_pairs = 0
        while step < self.maxstep and (change_pairs > 0 or entire_set):  # 当搜寻全部样本，依然没有改变，则停止迭代
            step += 1
            change_pairs = 0
            if entire_set:  # 搜索整个样本集
                for ind1 in range(self.N):
                    if not self.satisfy_kkt(y_data[ind1], self.err_arr[ind1], self.alpha_arr[ind1]):
                        ind2 = self.select_second_alpha(ind1)
                        change_pairs += self.update(ind1, ind2)
            else:  # 搜索间隔边界上的支持向量(bound_search)
                bound_inds = np.where((0 < self.alpha_arr) & (self.alpha_arr < self.C))[0]
                for ind1 in bound_inds:
                    if not self.satisfy_kkt(y_data[ind1], self.err_arr[ind1], self.alpha_arr[ind1]):
                        ind2 = self.select_second_alpha(ind1)
                        change_pairs += self.update(ind1, ind2)
            if entire_set:  # 当前是对整个数据集进行搜索，则下一次搜索间隔边界上的支持向量
                entire_set = False
            elif change_pairs == 0:
                entire_set = True  # 当前是对间隔边界上的支持向量进行搜索，若未发生任何改变，则下一次搜索整个数据集
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
    # xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
    #     #                      np.linspace(-3, 3, 500))
    #     # np.random.seed(0)
    #     # X_data = np.random.randn(500, 2)
    #     # Y = np.logical_xor(X_data[:, 0] > 0, X_data[:, 1] > 0)
    #     # y = []
    #     # for i in Y:
    #     #     if i:
    #     #         y.append(1)
    #     #     else:
    #     #         y.append(-1)
    #     # y_data = np.array(y)
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
        S = SVM(kernel_option=False, maxstep=1000, epsilon=1e-6, C=1.0)
        S.fit(X_train, y_train)
        score = 0
        for X, y in zip(X_test, y_test):
            if S.predict(X) == y:
                score += 1
        print(score / len(y_test))
