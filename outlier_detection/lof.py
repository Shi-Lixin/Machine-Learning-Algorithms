# LOF异常值检测算法
from scipy.spatial.distance import cdist
import numpy as np


class LOF:
    def __init__(self, data, k, epsilon=1.0):
        self.data = data
        self.k = k
        self.epsilon = epsilon
        self.N = self.data.shape[0]

    def get_dist(self):
        # 计算欧式距离矩阵
        return cdist(self.data, self.data)

    def _kdist(self, arr):
        # 计算k距离
        inds_sort = np.argsort(arr)
        neighbor_ind = inds_sort[1:self.k + 1]  # 邻域内点索引
        return neighbor_ind, arr[neighbor_ind[-1]]

    def get_rdist(self):
        # 计算可达距离
        dist = self.get_dist()
        nei_kdist = np.apply_along_axis(self._kdist, 1, dist)
        nei_inds, kdist = zip(*nei_kdist)
        for i, k in enumerate(kdist):
            ind = np.where(dist[i] < k)  # 实际距离小于k距离，则可达距离为k距离
            dist[i][ind] = k
        return nei_inds, dist

    def get_lrd(self, nei_inds, rdist):
        # 计算局部可达密度
        lrd = np.zeros(self.N)
        for i, inds in enumerate(nei_inds):
            s = 0
            for j in inds:
                s += rdist[j, i]
            lrd[i] = self.k / s
        return lrd

    def run(self):
        # 计算局部离群因子
        nei_inds, rdist = self.get_rdist()
        lrd = self.get_lrd(nei_inds, rdist)
        score = np.zeros(self.N)
        for i, inds in enumerate(nei_inds):
            lrd_nei = sum(lrd[inds])
            score[i] = lrd_nei / self.k / lrd[i]

        return score, np.where(score > self.epsilon)[0]


if __name__ == '__main__':
    np.random.seed(42)
    X_inliers = 0.3 * np.random.randn(100, 2)
    X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    data = np.r_[X_inliers, X_outliers]

    lof = LOF(data, 5, epsilon=1.2)
    score, out_ind = lof.run()
    outliers = data[out_ind]

    import matplotlib.pyplot as plt

    plt.scatter(data[:, 0], data[:, 1], color='b')
    plt.scatter(outliers[:, 0], outliers[:, 1], color='r')
    plt.show()
