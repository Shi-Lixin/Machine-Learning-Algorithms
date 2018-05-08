"""
高斯混合模型+EM算法
以alpha(k)的概率选择第k个高斯模型，再以该高斯模型概率分布产生数据。其中alpha(k)就是隐变量
"""
import numpy as np
import math
import copy
from collections import defaultdict
from sklearn.cluster import KMeans


class GEM:
    def __init__(self, maxstep=1000, epsilon=1e-3, K=4):
        self.maxstep = maxstep
        self.epsilon = epsilon
        self.K = K  # 混合模型中的分模型的个数

        self.alpha = None  # 每个分模型前系数
        self.mu = None  # 每个分模型的均值向量
        self.sigma = None  # 每个分模型的协方差
        self.gamma_all_final = None  # 存储最终的每个样本对分模型的响应度，用于最终的聚类

        self.D = None  # 输入数据的维度
        self.N = None  # 输入数据总量

    def inin_param(self, data):
        # 初始化参数
        self.D = data.shape[1]
        self.N = data.shape[0]
        self.init_param_helper(data)
        return

    def init_param_helper(self, data):
        # KMeans初始化模型参数
        KMEANS = KMeans(n_clusters=self.K).fit(data)
        clusters = defaultdict(list)
        for ind, label in enumerate(KMEANS.labels_):
            clusters[label].append(ind)
        mu = []
        alpha = []
        sigma = []
        for inds in clusters.values():
            partial_data = data[inds]
            mu.append(partial_data.mean(axis=0))  # 分模型的均值向量
            alpha.append(len(inds) / self.N)  # 权重
            sigma.append(np.cov(partial_data.T))  # 协方差,D个维度间的协方差
        self.mu = np.array(mu)
        self.alpha = np.array(alpha)
        self.sigma = np.array(sigma)
        return

    def _phi(self, y, mu, sigma):
        # 获取分模型的概率
        s1 = 1.0 / math.sqrt(np.linalg.det(sigma))
        s2 = np.linalg.inv(sigma)  # d*d
        delta = np.array([y - mu])  # 1*d
        return s1 * math.exp(-1.0 / 2 * delta @ s2 @ delta.T)

    def fit(self, data):
        # 迭代训练
        self.inin_param(data)
        step = 0
        gamma_all_arr = None
        while step < self.maxstep:
            step += 1
            old_alpha = copy.copy(self.alpha)
            # E步
            gamma_all = []
            for j in range(self.N):
                gamma_j = []    # 依次求每个样本对K个分模型的响应度

                for k in range(self.K):
                    gamma_j.append(self.alpha[k] * self._phi(data[j], self.mu[k], self.sigma[k]))

                s = sum(gamma_j)
                gamma_j = [item/s for item in gamma_j]
                gamma_all.append(gamma_j)

            gamma_all_arr = np.array(gamma_all)
            # M步
            for k in range(self.K):
                gamma_k = gamma_all_arr[:, k]
                SUM = np.sum(gamma_k)
                # 更新权重
                self.alpha[k] = SUM / self.N  # 更新权重
                # 更新均值向量
                new_mu = sum([gamma * y for gamma, y in zip(gamma_k, data)]) / SUM  # 1*d
                self.mu[k] = new_mu
                # 更新协方差阵
                delta_ = data - new_mu   # n*d
                self.sigma[k] = sum([gamma * (np.outer(np.transpose([delta]), delta)) for gamma, delta in zip(gamma_k, delta_)]) / SUM  # d*d
            alpha_delta = self.alpha - old_alpha
            if np.linalg.norm(alpha_delta, 1) < self.epsilon:
                break
        self.gamma_all_final = gamma_all_arr
        return

    def predict(self):
        cluster = defaultdict(list)
        for j in range(self.N):
            max_ind = np.argmax(self.gamma_all_final[j])
            cluster[max_ind].append(j)
        return cluster

if __name__ == '__main__':

    def generate_data(N=500):
        X = np.zeros((N, 2))  # N*2, 初始化X
        mu = np.array([[5, 35], [20, 40], [20, 35], [45, 15]])
        sigma = np.array([[30, 0], [0, 25]])
        for i in range(N): # alpha_list=[0.3, 0.2, 0.3, 0.2]
            prob = np.random.random(1)
            if prob < 0.1:  # 生成0-1之间随机数
                X[i, :] = np.random.multivariate_normal(mu[0], sigma, 1)  # 用第一个高斯模型生成2维数据
            elif 0.1 <= prob < 0.3:
                X[i, :] = np.random.multivariate_normal(mu[1], sigma, 1)  # 用第二个高斯模型生成2维数据
            elif 0.3 <= prob < 0.6:
                X[i, :] = np.random.multivariate_normal(mu[2], sigma, 1)  # 用第三个高斯模型生成2维数据
            else:
                X[i, :] = np.random.multivariate_normal(mu[3], sigma, 1)  # 用第四个高斯模型生成2维数据
        return X


    data = generate_data()
    gem = GEM()
    gem.fit(data)
    # print(gem.alpha, '\n', gem.sigma, '\n', gem.mu)
    cluster = gem.predict()

    import matplotlib.pyplot as plt
    from itertools import cycle
    colors = cycle('grbk')
    for color, inds in zip(colors, cluster.values()):
        partial_data = data[inds]
        plt.scatter(partial_data[:,0], partial_data[:, 1], edgecolors=color)
    plt.show()