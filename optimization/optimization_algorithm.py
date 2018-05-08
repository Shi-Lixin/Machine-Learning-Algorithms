"""
优化算法的核心两点：1.寻找下降方向  2.一维搜索下降的步长
梯度下降算法以梯度下降的负方向为下降方向，学习率乘以梯度的模为步长
牛顿法以海塞矩阵逆矩阵与梯度乘积为下降方向（此时学习率恒为1）
拟牛顿法同牛顿法，仅仅以B代替海塞矩阵

一维不精确搜索：Wolfe和Armijo确定搜索步长（通常应用于拟牛顿法）
Armijo: f(Xk + alpha * Dk) - f(Xk) <= rho * alpha * Grad.T * Dk, alpha=beta*gamma**m(m为非负整数，beta>0)
Goldstein(需要联合Armijo): f(Xk + alpha * Dk) - f(Xk) >= (1-rho) * alpha * Grad.T * Dk
Wolfe-Powell(需要联合Armijo):  Grad(k+1).T * Dk >= sigma * Grad(k).T * Dk, rho<sigma<1

以线性回归为例：
平方误差损失函数的最小化的优化问题
J(X)=1/2m(f(X)-y)**2, f(X)=wX,m为样本总数， 最优化函数J(X)对w的梯度为X(wX-y)
"""

import numpy as np
from numpy.linalg import norm, inv


def BGD(data, epsilon=1e-2, maxstep=1000, alpha=0.1):
    # 批量梯度下降算法
    i = 0
    X = data[:, :-1]  # n*d
    y = data[:, -1]  # n*1
    w = np.ones(X.shape[1])  # 初始化权值 1*d
    while i < maxstep:
        Grad = (w @ X.T - y.T) @ X / X.shape[0]  # 1*d 计算梯度
        if norm(Grad) < epsilon:  # 如果梯度的第二范数小于阈值，则终止训练
            break
        w += -alpha * Grad  # 负梯度方向下降，更新自变量，即权值w
        i += 1
    return w


def SGD(data, epsilon=1e-2, maxstep=1000, alpha=0.1):
    # 随机梯度下降
    w = np.ones(data.shape[1] - 1)  # 初始化权值 1*d
    i = 0
    over = False
    while i < maxstep and not over:
        np.random.shuffle(data)  # 打乱数据集顺序
        for sample in data:
            Grad = (w * sample[:-1] - sample[-1]) * sample[-1]
            if norm(Grad) < epsilon:
                over = True
                break
            w += -alpha * Grad
    return w


def MBGD(data, epsilon=1e-2, maxstep=1000, alpha=0.1, bsize=4):
    # 小批量梯度下降
    w = np.ones(data.shape[1] - 1)  # 初始化权值 1*d
    i = 0
    over = False
    while i < maxstep and not over:
        np.random.shuffle(data)  # 打乱数据集顺序
        bathes = [data[(i - 1) * bsize:i * bsize, :] for i in range(1, data.shape[0] // bsize)]  # 将原数据集切分成小块
        for batch in bathes:
            X = batch[:, :-1]
            y = batch[:, -1]
            Grad = (w @ X.T - y.T) @ X / bsize
            if norm(Grad) < epsilon:
                over = True
                break
            w += -alpha * Grad
    return w


def NM(data, epsilon=1e-2, maxstep=1000):
    # 牛顿法
    i = 0
    X = data[:, :-1]  # n*d
    x1 = data[:, 0]  # n*1
    x2 = data[:, 1]  # n*1
    y = data[:, -1]  # n*1
    m = data.shape[0]  # 样本总数
    w = np.ones(X.shape[1])  # 初始化权值 1*d
    while i < maxstep:
        err = w @ X.T - y.T  # 1*n
        Grad = err @ X / m  # 计算梯度
        if norm(Grad) < epsilon:  # 如果梯度的第二范数小于阈值，则终止训练
            break
        H12 = 2 * x1.T @ x2
        H = np.array([[2 * err @ x1, H12], [H12, 2 * err @ x2]])  # 计算海塞矩阵
        alpha = inv(H)  # d*d 计算海塞矩阵的逆矩阵，以此作为学习率
        w -= Grad @ alpha  # 负梯度方向下降，更新自变量，即权值w
        i += 1
    return w


def BFGS(data, epsilon=1e-4, maxstep=1000):
    # 拟牛顿算法
    i = 0
    X = data[:, :-1]  # n*d
    y = data[:, -1]  # n*1
    N = data.shape[0]
    w = np.ones(X.shape[1])  # 初始化权值 1*d
    B = np.eye(2)  # 初始化B
    while i < maxstep:
        err = w @ X.T - y.T
        fx = err @ err.T / 2 / N
        Grad = err @ X / N  # 1*d 计算梯度
        if norm(Grad) < epsilon:  # 如果梯度的第二范数小于阈值，则终止训练
            break
        Dk = -Grad @ inv(B)  # 1*d, 下降方向
        alpha, w = WPSearch(w, Dk, fx, Grad, X, y, N)  # 下降步长以及更新w, 注意弱用WP规则，还可以返回新的梯度，减少重复计算
        delta = alpha * Dk  # 1*d 自变量w的增量
        yk = B @ delta.T  # d*1, 更新yk
        B = B + yk @ yk.T / (delta @ yk) - B @ delta.T @ (delta @ B) / (delta @ B @ delta.T)  # 更新B
    return w


def WPSearch(w, Dk, fx, Grad, X, y, N, sigma=0.75, gamma=0.5, rho=1e-4, beta=1, maxm=100):
    # 基于Wolfe-Powell条件的不精确一维搜索
    assert ((rho < 1.0 / 2) and (rho > 0))
    assert ((gamma < 1.0) and (gamma > 0.0))
    assert ((sigma > rho) and (sigma < 1))
    assert (beta > 0)
    m = 0
    w_new = None
    alpha = None
    while m < maxm:
        alpha = beta * gamma ** m  # 一维搜索合适的m,进而更新alpha
        w_new = w + alpha * Dk
        err = w_new @ X.T - y.T
        fx_new = err @ err.T / 2 / N  # 下降后的函数值
        diff_val = fx_new - fx  # 下降量
        exp_diff = alpha * Grad @ Dk.T
        Grad_new = err @ X / N  # 更新w后的梯度
        if (diff_val <= rho * exp_diff) and (Grad_new @ Dk.T >= sigma * Grad @ Dk.T):
            break
        m += 1
    return alpha, w_new


def AGSearch(w, Dk, fx, Grad, X, y, N, gamma=0.5, rho=1e-4, beta=1, maxm=100):
    # 基于Armijo-Goldstein条件的不精确一维搜索
    assert ((rho < 1.0 / 2) and (rho > 0))
    assert ((gamma < 1.0) and (gamma > 0.0))
    assert (beta > 0)
    m = 0
    alpha = None
    w_new = None
    while m < maxm:
        alpha = beta * gamma ** m  # 一维搜索合适的m,进而更新alpha
        w_new = w + alpha * Dk
        err = w_new @ X.T - y.T
        fx_new = err @ err.T / 2 / N  # 下降后的函数值
        diff_val = fx_new - fx  # 下降量
        exp_diff = alpha * Grad @ Dk.T
        if (diff_val <= rho * exp_diff) and (diff_val >= (1 - rho) * exp_diff):
            break
        m += 1
    return alpha, w_new


def predict(X, w):
    res = []
    for sample in X:
        res.append(w @ sample.T)
    return res


if __name__ == '__main__':
    data = np.array([[1.1, 1.5, 2.5],
                     [1.3, 1.9, 3.2],
                     [1.5, 2.3, 3.9],
                     [1.7, 2.7, 4.6],
                     [1.9, 3.1, 5.3],
                     [2.1, 3.5, 6.0],
                     [2.3, 3.9, 6.7],
                     [2.5, 4.3, 7.4],
                     [2.7, 4.7, 8.1],
                     [2.9, 5.1, 8.8]])
    test_data = np.array([[3.1, 5.5, 9.5],
                          [3.3, 5.9, 10.2],
                          [3.5, 6.3, 10.9],
                          [3.7, 6.7, 11.6],
                          [3.9, 7.1, 12.3]])
    w = BFGS(data, epsilon=1e-6)
    print(predict(test_data[:, :-1], w))
