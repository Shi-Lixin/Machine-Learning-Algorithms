"""
应用最小二乘法进行一维线性回归
求导直接求得系数
不加正则化项： w = 1/(X.T@X)@X.T@y
加L2正则化项： w = 1/(X.T@X+alpha*E)@X.T@y
"""
import numpy as np
from numpy.linalg import inv


def LR_regression(data, alpha=0.0):
    X = data[:, :-1]  # n*d
    X = np.column_stack((X, np.ones(X.shape[0])))  # 末尾添加一列，元素全部为1
    y = data[:, -1]  # n*1
    w = (inv(X.T@X+alpha*np.eye(X.shape[1]))@X.T@y).T  # 1*d, 每个xi前的系数，也可以理解为每个特征的权重
    return w

def predict(X, w):
    X = np.column_stack((X, np.ones(X.shape[0])))
    return w@X.T

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = []
    with open('./data/data.txt') as fp:
        for line in fp:
            tmp =[]
            for item in line.strip().split('\t'):
                tmp.append(float(item))
            data.append(tmp)
    data = np.array(data)
    w = LR_regression(data)
    res = predict(data[:,0], w)
    plt.scatter(data[:,0], data[:,1], marker='o')
    plt.plot(data[:,0],res, color='r')
    plt.show()