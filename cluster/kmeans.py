"""
K均值聚类算法
给定初始簇的个数，迭代更改样本与簇的隶属关系，更新簇的中心为样本的均值
"""
from collections import defaultdict
import numpy as np
import copy


class KMEANS:
    def __init__(self, n_cluster, epsilon=1e-3, maxstep=2000):
        self.n_cluster = n_cluster
        self.epsilon = epsilon
        self.maxstep = maxstep
        self.N = None
        self.centers = None
        self.cluster = defaultdict(list)

    def init_param(self, data):
        # 初始化参数, 包括初始化簇中心
        self.N = data.shape[0]
        random_ind = np.random.choice(self.N, size=self.n_cluster)
        self.centers = [data[i] for i in random_ind]  # list存储中心点坐标数组
        for ind, p in enumerate(data):
            self.cluster[self.mark(p)].append(ind)
        return

    def _cal_dist(self, center, p):
        # 计算点到簇中心的距离平方
        return sum([(i - j) ** 2 for i, j in zip(center, p)])

    def mark(self, p):
        # 计算样本点到每个簇中心的距离，选取最小的簇
        dists = []
        for center in self.centers:
            dists.append(self._cal_dist(center, p))
        return dists.index(min(dists))

    def update_center(self, data):
        # 更新簇的中心坐标
        for label, inds in self.cluster.items():
            self.centers[label] = np.mean(data[inds], axis=0)
        return

    def divide(self, data):
        # 重新对样本聚类
        tmp_cluster = copy.deepcopy(self.cluster)  # 迭代过程中，字典长度不能发生改变，故deepcopy
        for label, inds in tmp_cluster.items():
            for i in inds:
                new_label = self.mark(data[i])
                if new_label == label:  # 若类标记不变，跳过
                    continue
                else:
                    self.cluster[label].remove(i)
                    self.cluster[new_label].append(i)
        return

    def cal_err(self, data):
        # 计算MSE
        mse = 0
        for label, inds in self.cluster.items():
            partial_data = data[inds]
            for p in partial_data:
                mse += self._cal_dist(self.centers[label], p)
        return mse / self.N

    def fit(self, data):
        self.init_param(data)
        step = 0
        while step < self.maxstep:
            step += 1
            self.update_center(data)
            self.divide(data)
            err = self.cal_err(data)
            if err < self.epsilon:
                break
        return


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from itertools import cycle
    import matplotlib.pyplot as plt

    data, label = make_blobs(centers=4, cluster_std=0.5)
    km = KMEANS(4)
    km.fit(data)
    cluster = km.cluster
    centers = np.array(km.centers)

    def visualize(data, cluster, centers):
        color = 'bgrym'
        for col, inds in zip(cycle(color), cluster.values()):
            partial_data = data[inds]
            plt.scatter(partial_data[:, 0], partial_data[:, 1], color=col)
        plt.scatter(centers[:, 0], centers[:, 1], color='k', marker='*', s=100)
        plt.show()
        return


    visualize(data, cluster, centers)
