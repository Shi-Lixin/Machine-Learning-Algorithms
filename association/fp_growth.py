"""
FP树增长算法发现频繁项集
"""
from collections import defaultdict, Counter, deque
import math
import copy


class node:
    def __init__(self, item, count, parent):  # 本程序将节点之间的链接信息存储到项头表中，后续可遍历项头表添加该属性
        self.item = item  # 该节点的项
        self.count = count  # 项的计数
        self.parent = parent  # 该节点父节点的id
        self.children = []  # 该节点的子节点的list


class FP:
    def __init__(self, minsup=0.5):
        self.minsup = minsup
        self.minsup_num = None  # 支持度计数

        self.N = None
        self.item_head = defaultdict(list)  # 项头表
        self.fre_one_itemset = defaultdict(lambda: 0)  # 频繁一项集，值为支持度
        self.sort_rules = None  # 项头表中的项排序规则，按照支持度从大到小有序排列
        self.tree = defaultdict()  # fp树， 键为节点的id, 值为node
        self.max_node_id = 0  # 当前树中最大的node_id, 用于插入新节点时，新建node_id
        self.fre_itemsets = []  # 频繁项集
        self.fre_itemsets_sups = []  # 频繁项集的支持度计数

    def init_param(self, data):
        self.N = len(data)
        self.minsup_num = math.ceil(self.minsup * self.N)
        self.get_fre_one_itemset(data)
        self.build_tree(data)
        return

    def get_fre_one_itemset(self, data):
        # 获取频繁1项，并排序，第一次扫描数据集
        c = Counter()
        for t in data:
            c += Counter(t)
        for key, val in c.items():
            if val >= self.minsup_num:
                self.fre_one_itemset[key] = val
        sort_keys = sorted(self.fre_one_itemset, key=self.fre_one_itemset.get, reverse=True)
        self.sort_rules = {k: i for i, k in enumerate(sort_keys)}  # 频繁一项按照支持度降低的顺序排列，构建排序规则
        return

    def insert_item(self, parent, item):
        # 将事务中的项插入到FP树中，并返回插入节点的id
        children = self.tree[parent].children
        for child_id in children:
            child_node = self.tree[child_id]
            if child_node.item == item:
                self.tree[child_id].count += 1
                next_node_id = child_id
                break
        else:  # 循环正常结束，表明当前父节点的子节点中没有项与之匹配，所以新建子节点，更新项头表和树
            self.max_node_id += 1
            next_node_id = copy.copy(self.max_node_id)  # 注意self.max_node_id 是可变的，引用时需要copy
            self.tree[next_node_id] = node(item=item, count=1, parent=parent)  # 更新树，添加节点
            self.tree[parent].children.append(next_node_id)  # 更新父节点的孩子列表
            self.item_head[item].append(next_node_id)  # 更新项头表
        return next_node_id

    def build_tree(self, data):
        # 构建项头表以及FP树， 第二次扫描数据集
        one_itemset = set(self.fre_one_itemset.keys())
        self.tree[0] = node(item=None, count=0, parent=-1)
        for t in data:
            t = list(set(t) & one_itemset)  # 去除该事务中非频繁项
            if len(t) > 0:
                t = sorted(t, key=lambda x: self.sort_rules[x])  # 按照项的频繁程度从大到小排序
                parent = 0  # 每个事务都是从树根开始插起
                for item in t:
                    parent = self.insert_item(parent, item)  # 将排序后的事务中每个项依次插入FP树
        return

    def get_path(self, pre_tree, condition_tree, node_id, suffix_items_count):
        # 根据后缀的某个叶节点的父节点出发，选取出路径，并更新计数。suffix_item_count为后缀的计数
        if node_id == 0:
            return
        else:
            if node_id not in condition_tree.keys():
                current_node = copy.deepcopy(pre_tree[node_id])
                current_node.count = suffix_items_count  # 更新计数
                condition_tree[node_id] = current_node

            else:  # 若叶节点有多个，则路径可能有重复，计数叠加
                condition_tree[node_id].count += suffix_items_count
            node_id = condition_tree[node_id].parent
            self.get_path(pre_tree, condition_tree, node_id, suffix_items_count)  # 递归构建路径
            return

    def get_condition_tree(self, pre_tree, suffix_items_ids):
        # 构建后缀为一个项的条件模式基。可能对应多个叶节点，综合后缀的各个叶节点的路径
        condition_tree = defaultdict()  # 字典存储条件FP树，值为父节点
        for suffix_id in suffix_items_ids:  # 从各个后缀叶节点出发，综合各条路径形成条件FP树
            suffix_items_count = copy.copy(pre_tree[suffix_id].count)  # 叶节点计数
            node_id = pre_tree[suffix_id].parent  # 注意条件FP树不包括后缀
            if node_id == 0:
                continue
            self.get_path(pre_tree, condition_tree, node_id, suffix_items_count)
        return condition_tree

    def extract_suffix_set(self, condition_tree, suffix_items):
        # 根据条件模式基，提取频繁项集, suffix_item为该条件模式基对应的后缀
        # 返回新的后缀，以及新添加项(将作为下轮的叶节点)的id
        new_suffix_items_list = []  # 后缀中添加的新项
        new_item_head = defaultdict(list)  # 基于当前的条件FP树，更新项头表， 新添加的后缀项
        item_sup_dict = defaultdict(int)
        for key, val in condition_tree.items():
            item_sup_dict[val.item] += val.count  # 对项出现次数进行统计
            new_item_head[val.item].append(key)

        for item, sup in item_sup_dict.items():
            if sup >= self.minsup_num:  # 若条件FP树中某个项是频繁的，则添加到后缀中
                current_item_set = [item] + suffix_items
                self.fre_itemsets.append(current_item_set)
                self.fre_itemsets_sups.append(sup)
                new_suffix_items_list.append(current_item_set)
            else:
                new_item_head.pop(item)
        return new_suffix_items_list, new_item_head.values()

    def get_fre_set(self, data):
        # 构建以每个频繁1项为后缀的频繁项集
        self.init_param(data)
        suffix_items_list = []
        suffix_items_id_list = []
        for key, val in self.fre_one_itemset.items():
            suffix_items = [key]
            suffix_items_list.append(suffix_items)
            suffix_items_id_list.append(self.item_head[key])
            self.fre_itemsets.append(suffix_items)
            self.fre_itemsets_sups.append(val)
        pre_tree = copy.deepcopy(self.tree)  # pre_tree 是尚未去除任何后缀的前驱，若其叶节点的项有多种，则可以形成多种条件FP树
        self.dfs_search(pre_tree, suffix_items_list, suffix_items_id_list)
        return

    def bfs_search(self, pre_tree, suffix_items_list, suffix_items_id_list):
        # 宽度优先，递增构建频繁k项集
        q = deque()
        q.appendleft((pre_tree, suffix_items_list, suffix_items_id_list))
        while len(q) > 0:
            param_tuple = q.pop()
            pre_tree = param_tuple[0]
            for suffix_items, suffix_items_ids in zip(param_tuple[1], param_tuple[2]):
                condition_tree = self.get_condition_tree(pre_tree, suffix_items_ids)
                new_suffix_items_list, new_suffix_items_id_list = self.extract_suffix_set(condition_tree, suffix_items)
                if new_suffix_items_list:
                    q.appendleft(
                        (condition_tree, new_suffix_items_list, new_suffix_items_id_list))  # 储存前驱，以及产生该前驱的后缀的信息
        return

    def dfs_search(self, pre_tree, suffix_items_list, suffix_items_id_list):
        # 深度优先，递归构建以某个项为后缀的频繁k项集
        for suffix_items, suffix_items_ids in zip(suffix_items_list, suffix_items_id_list):
            condition_tree = self.get_condition_tree(pre_tree, suffix_items_ids)
            new_suffix_items_list, new_suffix_items_id_list = self.extract_suffix_set(condition_tree, suffix_items)
            if new_suffix_items_list:  # 如果后缀有新的项添加进来，则继续深度搜索
                self.dfs_search(condition_tree, new_suffix_items_list, new_suffix_items_id_list)
        return


if __name__ == '__main__':
    data1 = [list('ABCEFO'), list('ACG'), list('ET'), list('ACDEG'), list('ACEGL'),
             list('EJ'), list('ABCEFP'), list('ACD'), list('ACEGM'), list('ACEGN')]
    data2 = [list('ab'), list('bcd'), list('acde'), list('ade'), list('abc'),
             list('abcd'), list('a'), list('abc'), list('abd'), list('bce')]
    data3 = [['r', 'z', 'h', 'j', 'p'], ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'], ['z'], ['r', 'x', 'n', 'o', 's'],
             ['y', 'r', 'x', 'z', 'q', 't', 'p'], ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]

    fp = FP(minsup=0.2)
    fp.get_fre_set(data2)

    for itemset, sup in zip(fp.fre_itemsets, fp.fre_itemsets_sups):
        print(itemset, sup)
