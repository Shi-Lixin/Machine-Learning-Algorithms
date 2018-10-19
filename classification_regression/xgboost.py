# 实现XGBoost回归, 以MSE损失函数为例
import numpy as np


class Node:
    def __init__(self, sp=None, left=None, right=None, w=None):
        self.sp = sp  # 非叶节点的切分，特征以及对应的特征下的值组成的元组
        self.left = left
        self.right = right
        self.w = w  # 叶节点权重，也即叶节点输出值

    def isLeaf(self):
        return self.w


class Tree:
    def __init__(self, _gamma, _lambda, max_depth):
        self._gamma = _gamma  # 正则化项中T前面的系数
        self._lambda = _lambda  # 正则化项w前面的系数
        self.max_depth = max_depth
        self.root = None

    def _candSplits(self, X_data):
        # 计算候选切分点
        splits = []
        for fea in range(X_data.shape[1]):
            for val in X_data[fea]:
                splits.append((fea, val))
        return splits

    def split(self, X_data, sp):
        # 劈裂数据集，返回左右子数据集索引
        lind = np.where(X_data[:, sp[0]] <= sp[1])[0]
        rind = list(set(range(X_data.shape[0])) - set(lind))
        return lind, rind

    def calWeight(self, garr, harr):
        # 计算叶节点权重，也即位于该节点上的样本预测值
        return - sum(garr) / (sum(harr) + self._lambda)

    def calObj(self, garr, harr):
        # 计算某个叶节点的目标(损失)函数值
        return (-1.0 / 2) * sum(garr) ** 2 / (sum(harr) + self._lambda) + self._gamma

    def getBestSplit(self, X_data, garr, harr, splits):
        # 搜索最优切分点
        if not splits:
            return None
        else:
            bestSplit = None
            maxScore = -float('inf')
            score_pre = self.calObj(garr, harr)
            subinds = None
            for sp in splits:
                lind, rind = self.split(X_data, sp)
                if len(rind) < 2 or len(lind) < 2:
                    continue
                gl = garr[lind]
                gr = garr[rind]
                hl = harr[lind]
                hr = harr[rind]
                score = score_pre - self.calObj(gl, hl) - self.calObj(gr, hr)  # 切分后目标函数值下降量
                if score > maxScore:
                    maxScore = score
                    bestSplit = sp
                    subinds = (lind, rind)
            if maxScore < 0:  # pre-stopping
                return None
            else:
                return bestSplit, subinds

    def buildTree(self, X_data, garr, harr, splits, depth):
        # 递归构建树
        res = self.getBestSplit(X_data, garr, harr, splits)
        depth += 1
        if not res or depth >= self.max_depth:
            return Node(w=self.calWeight(garr, harr))
        bestSplit, subinds = res
        splits.remove(bestSplit)
        left = self.buildTree(X_data[subinds[0]], garr[subinds[0]], harr[subinds[0]], splits, depth)
        right = self.buildTree(X_data[subinds[1]], garr[subinds[1]], harr[subinds[1]], splits, depth)
        return Node(sp=bestSplit, right=right, left=left)

    def fit(self, X_data, garr, harr):
        splits = self._candSplits(X_data)
        self.root = self.buildTree(X_data, garr, harr, splits, 0)

    def predict(self, x):
        def helper(currentNode):
            if currentNode.isLeaf():
                return currentNode.w
            fea, val = currentNode.sp
            if x[fea] <= val:
                return helper(currentNode.left)
            else:
                return helper(currentNode.right)

        return helper(self.root)

    def _display(self):
        def helper(currentNode):
            if currentNode.isLeaf():
                print(currentNode.w)
            else:
                print(currentNode.sp)
            if currentNode.left:
                helper(currentNode.left)
            if currentNode.right:
                helper(currentNode.right)

        helper(self.root)


class Forest:
    def __init__(self, n_iter, _gamma, _lambda, max_depth, eta=1.0):
        self.n_iter = n_iter  # 迭代次数，即基本树的个数
        self._gamma = _gamma
        self._lambda = _lambda
        self.max_depth = max_depth  # 单颗基本树最大深度
        self.eta = eta  # 收缩系数, 默认1.0,即不收缩
        self.trees = []

    def calGrad(self, y_pred, y_data):
        # 计算一阶导数
        return 2 * (y_pred - y_data)

    def calHess(self, y_pred, y_data):
        # 计算二阶导数
        return 2 * np.ones_like(y_data)

    def fit(self, X_data, y_data):
        step = 0
        while step < self.n_iter:
            tree = Tree(self._gamma, self._lambda, self.max_depth)
            y_pred = self.predict(X_data)
            garr, harr = self.calGrad(y_pred, y_data), self.calHess(y_pred, y_data)
            tree.fit(X_data, garr, harr)
            self.trees.append(tree)
            step += 1

    def predict(self, X_data):
        if self.trees:
            y_pred = []
            for x in X_data:
                y_pred.append(self.eta * sum([tree.predict(x) for tree in self.trees]))
            return np.array(y_pred)
        else:
            return np.zeros(X_data.shape[0])


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt

    boston = load_boston()
    y = boston['target']
    X = boston['data']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    f = Forest(50, 0, 1.0, 4, eta=0.8)
    f.fit(X_train, y_train)
    y_pred = f.predict(X_test)
    print(mean_absolute_error(y_test, y_pred))
    plt.scatter(np.arange(y_pred.shape[0]), y_test - y_pred)
    plt.show()
