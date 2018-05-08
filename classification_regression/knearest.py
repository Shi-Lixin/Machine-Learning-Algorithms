from collections import Counter
from itertools import cycle


class node:
    def __init__(self, val, label, left=None, right=None, visit=False):
        self.val = val
        self.label = label
        self.left = left
        self.right = right
        self.visit = visit  # 用于回退时，父节点搜索另一个子节点


class heap:
    # 建立一个最大堆来保存最近的k个样本点
    def __init__(self):
        self.h = []

    def push(self, item):
        # 加入元素
        self.h.append(item)
        self._move_up(item)
        return

    def get(self):
        # 获取堆顶元素
        if self.h:
            return self.h[0]
        else:
            raise ValueError('the heap is empty')

    def delete(self):
        # 删除堆顶元素
        if self.h:
            last_item = self.h.pop()
            if self.h:
                self.h[0] = last_item
                self._move_down(last_item)
        else:
            raise ValueError('the heap is empty')

    def _move_up(self, item):
        # 将新加入的元素上移
        startpos = len(self.h) - 1
        pos = (startpos - 1) >> 1
        while pos >= 0:
            if item[1] > self.h[pos][1]:
                self.h[pos], self.h[startpos] = self.h[startpos], self.h[pos]
                startpos = pos
                pos = (pos - 1) >> 1
            else:
                break
        return

    def _move_down(self, item):
        # 删除堆顶元素，将末尾元素加入堆顶，重新调整堆
        pos = 0
        child_pos = 2 * pos + 1  # 暂设左树为预交换节点
        while child_pos < len(self.h):
            rightpos = 2 * pos + 2
            if rightpos < len(self.h) and self.h[rightpos][1] > self.h[child_pos][1]:
                child_pos = rightpos
            if item[1] < self.h[child_pos][1]:
                self.h[pos], self.h[child_pos] = self.h[child_pos], self.h[pos]
                pos = child_pos
                child_pos = 2 * pos + 1
                continue
            else:
                break
        return


def get_split(data, d):
    # 根据切分维度返回切分点索引以及，切分后的两个子数据集
    vector = data[:, d]
    median = int(len(vector) / 2)
    inds = np.argpartition(vector, median)
    left = inds[:median]
    right = inds[median + 1:]
    return inds[median], left, right


def build_tree(data):
    # 构建kd树, 存储索引
    dimension_cycle = cycle(range(data.shape[1] - 1))

    def helper(dataset):
        if dataset.shape[0] < 1:
            return
        d = next(dimension_cycle)
        split, left, right = get_split(dataset, d)
        left = helper(dataset[left])
        right = helper(dataset[right])
        return node(val=dataset[split][:-1], label=dataset[split][-1], left=left, right=right)

    return helper(data)


def cal_dis(node, X):
    # 计算输入X与节点之间的距离
    delta = node.val - X
    return delta @ delta


def add_node(res, current_node, X, k):
    # 检查是否加入当前节点为k近邻之一
    dis = cal_dis(current_node, X)
    if len(res.h) < k:
        res.push([current_node, dis])
    else:
        if res.get()[1] > dis:
            res.delete()
            res.push([current_node, dis])
    return


def check_cross(X, d, current_node, dis):
    # 判断超球体是否与分割平面相交
    plane_dis = (X[d] - current_node.val[d])**2
    if plane_dis < dis:
        return True
    else:
        return False


def search_tree(X, tree, k=3):
    # k近邻搜索, 最大堆存储k个最近的元素
    dimension_cycle = cycle(range(X.shape[0]))
    res = heap()

    def helper(current_node):  # 从叶节点开始回退搜索
        nonlocal res
        # 寻找叶节点
        if current_node.left is None and current_node.right is None:
            current_node.visit = True
            dis = cal_dis(current_node, X)
            res.push([current_node, dis])
            return current_node
        d = next(dimension_cycle)
        if X[d] < current_node.val[d]:
            if current_node.left is not None:
                helper(current_node.left)
        elif current_node.right is not None:
            helper(current_node.right)
        # 回退搜索
        current_node.visit = True
        add_node(res, current_node, X, k)  # 回退到父节点，并检查父节点
        if check_cross(X, d, current_node, res.get()[1]):  # 如果与分割平面相交，则搜索另一个子节点
            if current_node.left is not None and current_node.left.visit:  # 检查另一个子节点
                add_node(res, current_node.left, X, k)
            if current_node.right is not None and current_node.right.visit:
                add_node(res, current_node.right, X, k)
        return

    helper(tree)

    return res


class KNearest:
    def __init__(self, k=5):
        self.k = k

    def predict(self, X_data, y_data, X):
        data = np.hstack((X_data, np.transpose([y_data])))
        tree = build_tree(data)
        pred_res = []
        for x in X:
            res = search_tree(x, tree, self.k)
            klabel = []
            for item in res.h:
                klabel.append(item[0].label)
            c = Counter(klabel)
            pred_res.append(max(c, key=c.get))
        return pred_res


if __name__ == '__main__':
    import numpy as np

    # X_data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    # y_data = np.array([0, 0, 1, 1, 0, 1])
    # data = np.hstack((X_data, np.transpose([y_data])))
    # tree = build_tree(data)

    # def disp_tree(tree):
    #     # 打印树
    #     def disp_helper(current_node):
    #         # 前序遍历
    #         print(current_node.val, current_node.label)
    #         if current_node.left is not None:
    #             disp_helper(current_node.left)
    #         if current_node.right is not None:
    #             disp_helper(current_node.right)
    #         return
    #
    #     disp_helper(tree)
    #     return

    # disp_tree(tree)

    # res = search_tree(np.array([4, 3]), tree, k=3)
    # for node, dis in res:
    #     print(node.val, node.label, dis)

    from sklearn.datasets import make_blobs
    from machine_learning_algorithm.cross_validation import validate

    X_data, y_data = make_blobs(n_samples=200)
    g = validate(X_data, y_data, ratio=0.2)
    for item in g:
        X_data_train, y_data_train, X_data_test, y_data_test = item
        knn = KNearest()
        score = 0
        y_pred = knn.predict(X_data_train, y_data_train, X_data_test)
        for y_test, y_pred in zip(y_data_test, y_pred):
            if y_test == y_pred:
                score += 1
        print(score / len(y_data_test))
