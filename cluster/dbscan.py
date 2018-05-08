import numpy as np
from collections import defaultdict, deque
import math


class DBSCAN:
    def __init__(self):
        self.eps = None
        self.minpts = None
        self.N = None
        self.data = None
        self.pair_dis = None
        self.labels = None
        self.cluster_labels = None

    def init_param(self, data):
        # 初始化参数
        self.data = data
        self.N = data.shape[0]
        self.pair_dis = self.cal_pair_dis()
        self.cluster_labels = np.zeros(self.data.shape[0])  # 将所有点类标记设为 0
        self.cal_pair_dis()
        return

    def _cal_dis(self, p1, p2):
        dis = 0
        for i, j in zip(p1, p2):
            dis += (i - j) ** 2
        return math.sqrt(dis)

    def cal_pair_dis(self):
        # 获取每对点之间的距离
        pair_dis = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i + 1, self.N):
                pair_dis[i, j] = self._cal_dis(self.data[i], self.data[j])
                pair_dis[j, i] = pair_dis[i, j]
        return pair_dis

    def _cal_k_dis(self, k):
        # 计算每个点的k距离
        kdis = []
        for i in range(self.N):
            dis_arr = self.pair_dis[i, :]
            inds_sort = np.argsort(dis_arr)
            kdis.append(dis_arr[inds_sort[k + 1]])
        return kdis

    def graph2param(self, minpts):
        # 画出k距离图，决定参数eps, minpts
        kdis = self._cal_k_dis(minpts)
        kdis = sorted(kdis)
        plt.plot(kdis)
        plt.show()
        return

    def cal_eps_neighbors(self):
        # 计算某个点eps内距离点的集合
        if self.eps is None:
            raise ValueError('the eps is not set')
        neighs = defaultdict(list)
        for i in range(self.N):
            for ind, dis in enumerate(self.pair_dis[i]):
                if dis <= self.eps and i != ind:
                    neighs[i].append(ind)
        return neighs

    def mark_core(self, neighs):
        # 标记核心点
        if self.minpts is None:
            raise ValueError('the minpts is not set')
        core_points = []
        for key, val in neighs.items():
            if len(val) >= self.minpts:  # 近邻点数大于minpts则为核心点
                core_points.append(key)
        return core_points

    def fit(self):
        # 训练，对每个样本进行判别
        neighs = self.cal_eps_neighbors()
        core_points = self.mark_core(neighs)
        cluster_label = 0
        q = deque()
        for p in core_points:
            if not self.cluster_labels[p]:  # 若该核心点未被标记，则建立新簇， 簇标记加 1
                q.append(p)
                cluster_label += 1
                # 以当前核心点为出发点， 采用广度优先算法进行簇的扩展， 直到队列为空，则停止此次扩展
                while len(q) > 0:
                    p = q.pop()
                    self.cluster_labels[p] = cluster_label
                    for n in neighs[p]:
                        if not self.cluster_labels[n]:  # 邻域内的点未归类，则加入该簇
                            self.cluster_labels[n] = cluster_label
                            if n in core_points:  # 若邻域内存在未标记的核心点，则依据该核心点继续扩展簇（一定要未标记，否则造成死循环）
                                q.appendleft(n)

        return


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.datasets import make_blobs

    data, label = make_blobs(centers=5, cluster_std=1.5, random_state=5)
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.show()
    db = DBSCAN()
    db.init_param(data)
    # db.graph2param(5)
    db.eps = 2.0
    db.minpts = 5
    db.fit()


    def visualize(data, cluster_labels):
        cluster = defaultdict(list)
        for ind, label in enumerate(cluster_labels):
            cluster[label].append(ind)
        color = 'bgrym'
        for col, label in zip(cycle(color), cluster.keys()):
            if label == 0:
                col = 'k'
            partial_data = data[cluster[label]]
            plt.scatter(partial_data[:, 0], partial_data[:, 1], color=col)
        plt.show()
        return


    visualize(data, db.cluster_labels)
