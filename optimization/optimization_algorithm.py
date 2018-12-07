"""
优化算法的核心两点：1.寻找下降方向  2.一维搜索下降的步长
梯度下降算法以负梯度方向为下降方向
牛顿法以海塞矩阵逆矩阵与梯度乘积为下降方向（此时学习率恒为1）
拟牛顿法同牛顿法，仅仅以B代替海塞矩阵

一维不精确搜索：Wolfe和Armijo确定搜索步长（通常应用于拟牛顿法）
Armijo: f(Xk + eta * Dk) - f(Xk) <= rho * eta * Grad.T * Dk, eta=beta*gamma**m(m为非负整数，beta>0)
Goldstein(需要联合Armijo): f(Xk + eta * Dk) - f(Xk) >= (1-rho) * eta * Grad.T * Dk
Wolfe-Powell(需要联合Armijo):  Grad(k+1).T * Dk >= sigma * Grad(k).T * Dk, rho<sigma<1

以线性回归为例：
平方误差损失函数的最小化的优化问题
J(X)=1/2m(f(X)-y)**2, f(X)=wX,m为样本总数， 最优化函数J(X)对w的梯度为X(wX-y)

输入X:n*d, y:n*1
初始化W:d*1
"""

import numpy as np
from numpy.linalg import norm, inv
from itertools import cycle


def grad(X, y, W):
    return X.T @ (X @ W - y) / X.shape[0]


def initial(X, y, batch_size):
    # 小批量梯度下降初始化
    assert batch_size > 1
    W = np.ones((X.shape[1], 1))
    indx = np.random.permutation(np.arange(X.shape[0]))
    X, y = X[indx], y[indx]
    indx_cycle = cycle(np.arange(X.shape[0] // batch_size + 1) * batch_size)
    return W, indx_cycle


def bgd(X, y, epsilon=1e-2, max_iter=1000, eta=0.1):
    # 批量梯度下降算法
    i = 0
    W = np.ones((X.shape[1], 1))  # 初始化权值 d*1
    while i < max_iter:
        Grad = grad(X, y, W)  # 1*d 计算梯度
        if norm(Grad) < epsilon:  # 如果梯度的第二范数小于阈值，则终止训练
            break
        W -= eta * Grad  # 负梯度方向下降，更新自变量，即权值w
        i += 1

    return W


def sgd(X, y, epsilon=1e-2, max_iter=3600, eta=0.1):
    # 随机梯度下降
    i = 0
    W = np.ones((X.shape[1], 1))
    while i < max_iter:
        indx = np.random.randint(X.shape[0], size=1)  # 随机选择一个样本
        X_s, y_s = X[(indx)], y[(indx)]
        Grad = grad(X_s, y_s, W)
        if norm(Grad) < epsilon:
            break
        W -= eta * Grad
        i += 1
    return W


def msgd(X, y, epsilon=1e-2, max_iter=1000, eta=0.1, batch_size=4):
    # 小批量梯度下降
    W, indx_cycle = initial(X, y, batch_size)
    i = 0
    while i < max_iter:
        j = next(indx_cycle)
        X_s, y_s = X[j:j + batch_size], y[j:j + batch_size]
        Grad = grad(X_s, y_s, W)
        if norm(Grad) < epsilon:
            break
        W -= eta * Grad
        i += 1
    return W


def momentum(X, y, epsilon=1e-2, max_iter=3600, eta=0.1, batch_size=4, alpha=0.01, nesterov=False):
    # 随机梯度下降
    W, indx_cycle = initial(X, y, batch_size)
    i = 0
    v = np.zeros((X.shape[1], 1))  # 初始化动量
    while i < max_iter:
        j = next(indx_cycle)
        X_s, y_s = X[j:j + batch_size], y[j:j + batch_size]
        if nesterov:
            W += alpha * v
        Grad = grad(X_s, y_s, W)
        if norm(Grad) < epsilon:
            break
        v = alpha * v - eta * Grad
        W += v
        i += 1
    return W


def adag(X, y, epsilon=1e-2, max_iter=1000, eta=0.1, batch_size=4, eps_station=1e-10):
    # adaptive gradient descent自适应学习率梯度下降
    W, indx_cycle = initial(X, y, batch_size)
    r = np.zeros((X.shape[1], 1))
    i = 0
    while i < max_iter:
        j = next(indx_cycle)
        X_s, y_s = X[j:j + batch_size], y[j:j + batch_size]
        Grad = grad(X_s, y_s, W)
        if norm(Grad) < epsilon:
            break
        r += Grad ** 2
        W -= eta * Grad / (np.sqrt(r) + eps_station)
        i += 1
    return W


def rms_prop(X, y, epsilon=1e-2, max_iter=1000, eta=0.01, rho=0.9, batch_size=4, eps_station=1e-10):
    # 均方根反向传播算法
    W, indx_cycle = initial(X, y, batch_size)
    m = np.zeros((X.shape[1], 1))
    i = 0
    while i < max_iter:
        j = next(indx_cycle)
        X_s, y_s = X[j:j + batch_size], y[j:j + batch_size]
        Grad = grad(X_s, y_s, W)
        if norm(Grad) < epsilon:
            break
        m = rho * m + (1 - rho) * Grad ** 2
        W -= eta * Grad / (np.sqrt(m) + eps_station)
        i += 1
    return W


def adam(X, y, epsilon=1e-2, max_iter=1000, eta=0.01, beta1=0.9, beta2=0.999, batch_size=4, eps_station=1e-10):
    # adam算法
    W, indx_cycle = initial(X, y, batch_size)
    m = np.zeros((X.shape[1], 1))
    v = np.zeros((X.shape[1], 1))
    i = 0
    while i < max_iter:
        j = next(indx_cycle)
        X_s, y_s = X[j:j + batch_size], y[j:j + batch_size]
        Grad = grad(X_s, y_s, W)
        if norm(Grad) < epsilon:
            break
        m = beta1 * m + (1 - beta1) * Grad
        v = beta2 * v + (1 - beta2) * Grad ** 2
        m_bar = m / (1 - beta1 ** (i + 1))
        v_bar = v / (1 - beta2 ** (i + 1))
        W -= eta * m_bar / (np.sqrt(v_bar) + eps_station)
        i += 1
    return W


def newton(X, y, epsilon=1e-2, max_iter=1000):
    # 牛顿法
    i = 0
    W = np.ones((X.shape[1], 1))
    x1 = X[:, 0]  # 变量维度1,n维向量
    x2 = X[:, 1]
    while i < max_iter:
        err = X @ W - y  # n*1
        Grad = X.T @ err / X.shape[0]
        if norm(Grad) < epsilon:  # 如果梯度的第二范数小于阈值，则终止训练
            break
        err = err.reshape(-1)
        H12 = 2 * x1 @ x2
        H11 = 2 * err @ x1
        H22 = 2 * err @ x2
        H = np.array([[H11, H12], [H12, H22]])  # 计算海塞矩阵
        W -= inv(H) @ Grad  # 负梯度方向下降，更新自变量，即权值w
        i += 1
    return W


def bfgs(X, y, epsilon=1e-4, max_iter=1000):
    # 拟牛顿算法
    i = 0
    W = np.ones((X.shape[1], 1))
    N = X.shape[0]
    B = np.eye(X.shape[1])  # 初始化B d*d
    while i < max_iter:
        err = X @ W - y
        fx = err.T @ err / 2 / N
        Grad = X.T @ err / N  # d*1 计算梯度
        if norm(Grad) < epsilon:  # 如果梯度的第二范数小于阈值，则终止训练
            break
        Dk = - inv(B) @ Grad  # d*1, 下降方向
        eta, W = wp_search(W, Dk, fx, Grad, X, y, N)  # 下降步长以及更新w, 注意弱用WP规则，还可以返回新的梯度，减少重复计算
        delta = eta * Dk  # d*1 自变量w的增量
        yk = B @ delta  # d*1, 更新yk
        B = B + yk @ yk.T / (yk.T @ delta)[0] - B @ delta @ (delta.T @ B) / (delta.T @ B @ delta)[0]  # 更新B
    return W


def wp_search(W, Dk, fx, Grad, X, y, N, sigma=0.75, gamma=0.5, rho=1e-4, beta=1, maxm=100):
    # 基于Wolfe-Powell条件的不精确一维搜索
    assert ((rho < 1.0 / 2) and (rho > 0))
    assert ((gamma < 1.0) and (gamma > 0.0))
    assert ((sigma > rho) and (sigma < 1))
    assert (beta > 0)
    m = 0
    W_new = None
    eta = None
    while m < maxm:
        eta = beta * gamma ** m  # 一维搜索合适的m,进而更新eta
        W_new = W + eta * Dk
        err_new = X @ W_new - y
        fx_new = err_new.T @ err_new / 2 / N  # 下降后的函数值
        diff_val = fx_new - fx  # 下降量
        gdk = Grad.T @ Dk
        exp_diff = eta * gdk
        Grad_new = X.T @ err_new / N  # 更新w后的梯度
        if (diff_val[0] <= rho * exp_diff[0]) and (Grad_new.T @ Dk >= sigma * gdk):
            break
        m += 1
    return eta, W_new


def ag_search(W, Dk, fx, Grad, X, y, N, gamma=0.5, rho=1e-4, beta=1, maxm=100):
    # 基于Armijo-Goldstein条件的不精确一维搜索
    assert ((rho < 1.0 / 2) and (rho > 0))
    assert ((gamma < 1.0) and (gamma > 0.0))
    assert (beta > 0)
    m = 0
    eta = None
    W_new = None
    while m < maxm:
        eta = beta * gamma ** m  # 一维搜索合适的m,进而更新eta
        W_new = W + eta * Dk
        err_new = X @ W_new - y
        fx_new = err_new.T @ err_new / 2 / N  # 下降后的函数值
        diff_val = fx_new - fx  # 下降量
        exp_diff = eta * Grad.T @ Dk
        if (diff_val[0] <= rho * exp_diff[0]) and (diff_val[0] >= (1 - rho) * exp_diff[0]):
            break
        m += 1
    return eta, W_new


def predict(X, W):
    return X @ W


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error
    from pprint import pprint

    train_data = np.array([[1.1, 1.5, 2.5],
                           [1.3, 1.9, 3.2],
                           [1.5, 2.3, 3.9],
                           [1.7, 2.7, 4.6],
                           [1.9, 3.1, 5.3],
                           [2.1, 3.5, 6.0],
                           [2.3, 3.9, 6.7],
                           [2.5, 4.3, 7.4],
                           [2.7, 4.7, 8.1],
                           [2.9, 5.1, 8.8]])
    X_train, y_train = train_data[:, :-1], train_data[:, [-1]]
    test_data = np.array([[3.1, 5.5, 9.5],
                          [3.3, 5.9, 10.2],
                          [3.5, 6.3, 10.9],
                          [3.7, 6.7, 11.6],
                          [3.9, 7.1, 12.3]])
    X_test, y_test = test_data[:, :-1], test_data[:, [-1]]

    W = bfgs(X_train, y_train, epsilon=1e-5, max_iter=3000)
    y_pred = predict(X_test, W)
    pprint(W)
    print(y_pred, '\n', mean_squared_error(y_test, y_pred))
