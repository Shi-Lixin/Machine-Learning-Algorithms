"""
标签传播聚类算法, 典型的半监督学习算法
核心思想：相似的数据应该具有相同的标签，构建节点间的相似性矩阵（边的权重）
"""
import numpy as np


class LablePropagation:
    def __init__(self, epsilon=1e-3, maxstep=500, kernel_option='rbf', sigma=1.0, k=10):
        self.epsilon = epsilon
        self.maxstep = maxstep
        self.kernel_option = kernel_option
        self.sigma = sigma  # rbf 核函数的参数
        self.k = k  # knn 核函数参数

        self.T = None  # 未标记点间的转换矩阵
        self.Y = None  # 标签数矩阵
        self.Y_clamp = None  # 已知标签数据点的标签矩阵
        self.N = None
        self.labeled_inds = None  # 已知标签样本的索引
        self.labels = None

    def init_param(self, X_data, y_data):
        # 初始化参数
        self.N = X_data.shape[0]
        self.labeled_inds = np.where(y_data >= 0)[0]  # 未知标签设为-1
        n_class = len(np.unique(y_data[self.labeled_inds]))

        self.Y = np.zeros((self.N, n_class))
        for i in self.labeled_inds:
            self.Y[i][int(y_data[i])] = 1.0   # 哑编码，对应标签设为1

        self.Y_clamp = self.Y[self.labeled_inds]  # n*l
        self.T = self.cal_tran_mat(X_data)  # n*n
        return

    def cal_dis2(self, node1, node2):
        # 计算节点间的欧式距离平方
        return (node1 - node2) @ (node1 - node2)

    def cal_tran_mat(self, data):
        # 计算转换矩阵, 即构建图
        dis_mat = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dis_mat[i, j] = self.cal_dis2(data[i], data[j])
                dis_mat[j, i] = dis_mat[i, j]

        if self.kernel_option == 'rbf':
            assert (self.sigma is not None)
            T = np.exp(-dis_mat / self.sigma ** 2)
            normalizer = T.sum(axis=0)
            T = T / normalizer
        elif self.kernel_option == 'knn':
            assert (self.k is not None)
            T = np.zeros((self.N, self.N))
            for i in range(self.N):
                inds = np.argpartition(dis_mat[i], self.k + 1)[:self.k + 1]
                T[i][inds] = 1.0 / self.k  # 最近的k个拥有相同的权重
                T[i][i] = 0
        else:
            raise ValueError('kernel is not supported')
        return T

    def fit(self, X_data, y_data):
        # 训练主函数
        self.init_param(X_data, y_data)
        step = 0
        while step < self.maxstep:
            step += 1
            new_Y = self.T @ self.Y  # 更新标签矩阵
            new_Y[self.labeled_inds] = self.Y_clamp  # clamp
            if np.abs(new_Y - self.Y).sum() < self.epsilon:
                break
            self.Y = new_Y
        self.labels = np.argmax(self.Y, axis=1)
        return


if __name__ == '__main__':
    from sklearn.datasets import make_circles

    n_samples = 100
    X, y = make_circles(n_samples=n_samples, shuffle=False)
    outer, inner = 0, 1
    labels = -np.ones(n_samples)
    labels[0] = outer
    labels[-1] = inner
    LPA = LablePropagation(maxstep=1000, kernel_option='knn', k=2, sigma=0.07)
    LPA.fit(X, labels)
    labels = LPA.labels

    import matplotlib.pyplot as plt


    def visualize(data, labels):
        color = 'bg'
        unique_label = np.unique(labels)
        for col, label in zip(color, unique_label):
            partial_data = data[np.where(labels == label)]
            plt.scatter(partial_data[:, 0], partial_data[:, 1], color=col, alpha=1)
        plt.scatter(data[0, 0], data[0, 1], color='b', marker='*', s=200, alpha=0.5)  # outer
        plt.scatter(data[-1, 0], data[-1, 1], color='g', marker='*', s=200, alpha=0.5)  # inner
        plt.show()
        return


    visualize(X, labels)
