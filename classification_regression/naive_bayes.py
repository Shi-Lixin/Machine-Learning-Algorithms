"""
朴素贝叶斯分类算法
采用后验期望计参数，先验概率分布取均匀分布
"""

from collections import Counter, defaultdict
import numpy as np


class NBayes:
    def __init__(self, lambda_=1):
        self.lambda_ = lambda_  # 贝叶斯估计参数lambda
        self.p_prior = {}  # 模型的先验概率, 注意这里的先验概率不是指预先人为设定的先验概率，而是需要估计的P(y=Ck)
        self.p_condition = {}  # 模型的条件概率

    def fit(self, X_data, y_data):
        N = y_data.shape[0]
        # 后验期望估计P(y=Ck)的后验概率，设定先验概率为均匀分布
        c_y = Counter(y_data)
        K = len(c_y)
        for key, val in c_y.items():
            self.p_prior[key] = (val + self.lambda_) / (N + K * self.lambda_)
        # 后验期望估计P(Xd=a|y=Ck)的后验概率，同样先验概率为均匀分布
        for d in range(X_data.shape[1]):  # 对各个维度分别进行处理
            Xd_y = defaultdict(int)
            vector = X_data[:, d]
            Sd = len(np.unique(vector))
            for xd, y in zip(vector, y_data):   # 这里Xd仅考虑出现在数据集D中的情况，故即使用极大似然估计叶没有概率为0的情况
                Xd_y[(xd, y)] += 1
            for key, val in Xd_y.items():
                self.p_condition[(d, key[0], key[1])] = (val + self.lambda_) / (c_y[key[1]] + Sd * self.lambda_)
        return

    def predict(self, X):
        p_post = defaultdict()
        for y, py in self.p_prior.items():
            p_joint = py  # 联合概率分布
            for d, Xd in enumerate(X):
                p_joint *= self.p_condition[(d, Xd, y)]  # 条件独立性假设
            p_post[y] = p_joint  # 分母P(X)相同，故直接存储联合概率分布即可
        return max(p_post, key=p_post.get)


if __name__ == '__main__':
    data = np.array([[1, 0, -1], [1, 1, -1], [1, 1, 1], [1, 0, 1],
                     [1, 0, -1], [2, 0, -1], [2, 1, -1], [2, 1, 1],
                     [2, 2, 1], [2, 2, 1], [3, 2, 1], [3, 1, 1],
                     [3, 1, 1], [3, 2, 1], [3, 2, -1]])
    X_data = data[:, :-1]
    y_data = data[:, -1]
    clf = NBayes(lambda_=1)
    clf.fit(X_data, y_data)
    print(clf.p_prior, '\n', clf.p_condition)
    print(clf.predict(np.array([2, 0])))
