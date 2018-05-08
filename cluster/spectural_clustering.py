"""
谱聚类算法
核心思想：构建样本点的图，切分图，使得子图内权重最大，子图间权重最小
"""
import numpy as np
from kmeans import KMEANS


class Spectrum:
    def __init__(self, n_cluster, epsilon=1e-3, maxstep=1000, method='unnormalized',
                 criterion='gaussian', gamma=2.0, dis_epsilon=70, k=5):
        self.n_cluster = n_cluster
        self.epsilon = epsilon
        self.maxstep = maxstep
        self.method = method  # 本程序提供规范化以及非规范化的谱聚类算法
        self.criterion = criterion  # 相似性矩阵的构建方法
        self.gamma = gamma  # 高斯方法中的sigma参数
        self.dis_epsilon = dis_epsilon  # epsilon-近邻方法的参数
        self.k = k  # k近邻方法的参数

        self.W = None  # 图的相似性矩阵
        self.L = None  # 图的拉普拉斯矩阵
        self.L_norm = None  # 规范化后的拉普拉斯矩阵
        self.D = None  # 图的度矩阵
        self.cluster = None

        self.N = None

    def init_param(self, data):
        # 初始化参数
        self.N = data.shape[0]
        dis_mat = self.cal_dis_mat(data)
        self.cal_weight_mat(dis_mat)
        self.D = np.diag(self.W.sum(axis=1))
        self.L = self.D - self.W
        return

    def cal_dis_mat(self, data):
        # 计算距离平方的矩阵
        dis_mat = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dis_mat[i, j] = (data[i] - data[j]) @ (data[i] - data[j])
                dis_mat[j, i] = dis_mat[i, j]
        return dis_mat

    def cal_weight_mat(self, dis_mat):
        # 计算相似性矩阵
        if self.criterion == 'gaussian':  # 适合于较小样本集
            if self.gamma is None:
                raise ValueError('gamma is not set')
            self.W = np.exp(-self.gamma * dis_mat)
        elif self.criterion == 'k_nearest':  # 适合于较大样本集
            if self.k is None or self.gamma is None:
                raise ValueError('k or gamma is not set')
            self.W = np.zeros((self.N, self.N))
            for i in range(self.N):
                inds = np.argpartition(dis_mat[i], self.k + 1)[:self.k + 1]  # 由于包括自身，所以+1
                tmp_w = np.exp(-self.gamma * dis_mat[i][inds])
                self.W[i][inds] = tmp_w
        elif self.criterion == 'eps_nearest':  # 适合于较大样本集
            if self.dis_epsilon is None:
                raise ValueError('epsilon is not set')
            self.W = np.zeros((self.N, self.N))
            for i in range(self.N):
                inds = np.where(dis_mat[i] < self.dis_epsilon)
                self.W[i][inds] = 1.0 / len(inds)
        else:
            raise ValueError('the criterion is not supported')
        return

    def fit(self, data):
        # 训练主函数
        self.init_param(data)
        if self.method == 'unnormalized':
            w, v = np.linalg.eig(self.L)
            inds = np.argsort(w)[:self.n_cluster]
            Vectors = v[:, inds]
        elif self.method == 'normalized':
            D = np.linalg.inv(np.sqrt(self.D))
            L = D @ self.L @ D
            w, v = np.linalg.eig(L)
            inds = np.argsort(w)[:self.n_cluster]
            Vectors = v[:, inds]
            normalizer = np.linalg.norm(Vectors, axis=1)
            normalizer = np.repeat(np.transpose([normalizer]), self.n_cluster, axis=1)
            Vectors = Vectors / normalizer
        else:
            raise ValueError('the method is not supported')
        km = KMEANS(self.n_cluster, self.epsilon, self.maxstep)
        km.fit(Vectors)
        self.cluster = km.cluster
        return


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from itertools import cycle
    import matplotlib.pyplot as plt

    data, label = make_blobs(centers=3, n_features=10, cluster_std=1.2, n_samples=500, random_state=1)
    sp = Spectrum(n_cluster=3, method='unnormalized', criterion='gaussian', gamma=0.1)
    sp.fit(data)
    cluster = sp.cluster

    # km = KMEANS(4)
    # km.fit(data)
    # cluster_km = km.cluster

    # def visualize(data, cluster):
    #     color = 'bgrym'
    #     for col, inds in zip(cycle(color), cluster.values()):
    #         partial_data = data[inds]
    #         plt.scatter(partial_data[:, 0], partial_data[:, 1], color=col)
    #     plt.show()
    #     return

    # visualize(data, cluster)

    def cal_err(data, cluster):
        # 计算MSE
        mse = 0
        for label, inds in cluster.items():
            partial_data = data[inds]
            center = partial_data.mean(axis=0)
            for p in partial_data:
                mse += (center - p) @ (center - p)
        return mse / data.shape[0]

    print(cal_err(data, cluster))
    # print(cal_err(data, cluster_km))