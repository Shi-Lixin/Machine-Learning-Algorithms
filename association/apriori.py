"""
Apriori关联分析算法
核心思想:先验原理
"""
from collections import Counter, defaultdict


class Apriori:
    def __init__(self, minsup, minconf):
        self.minsup = minsup
        self.minconf = minconf
        self.data = None
        self.N = None  # 购物篮数据的总数
        self.D = None  # 频繁项集的最大项个数
        self.fre_list = []  # 频繁项集，[[[],[]],[[],[]]]
        self.sup_list = []  # 存储每个频繁项的支持度
        self.fre_dict = defaultdict(lambda: 0)  # 键为频繁项集的tuple,值为支持度
        self.rules_dict = defaultdict(lambda: 0)  # 规则,键为规则前件和规则后件的tuple, 值为置信度

    def init_param(self, data):
        # 根据传入的数据初始化参数
        self.data = sorted(data)
        self.N = len(data)
        self.D = 0

        item_counter = Counter()
        for itemset in data:
            if len(itemset) > self.D:
                self.D = len(itemset)
            item_counter += Counter(itemset)
        itemset = sorted(item_counter)  # 保证有序
        c1 = []
        sup_c1 = []
        for item in itemset:
            sup = item_counter[item] / self.N
            if sup > self.minsup:
                c1.append([item])
                sup_c1.append(sup)

        self.fre_list.append(c1)
        self.sup_list.append(sup_c1)
        return

    def apriori_fre_itemset(self):
        # 使用Apriori算法获取频繁项集
        for i in range(1, self.D):  # 逐渐增加频繁项大小
            ck_1 = self.fre_list[i - 1]
            if len(ck_1) < 2:  # 若k-1频繁项集不足两个，则跳出循环
                break
            cand_ck_set = self.ck_itemset(i, ck_1)

            sup_ck = []
            ck = []
            for item in cand_ck_set:  # 计算ck的支持度
                sup = self.cal_sup(item)
                if sup > self.minsup:
                    ck.append(item)
                    sup_ck.append(sup)

            if len(ck) > 0:
                self.fre_list.append(ck)
                self.sup_list.append(sup_ck)

        for ck, sup_ck in zip(self.fre_list, self.sup_list):
            for itemset, sup in zip(ck, sup_ck):
                self.fre_dict[tuple(itemset)] = sup

        return

    def ck_itemset(self, ind, ck_1):
        # 根据k-1频繁项集产生k频繁项集, 产生候选然后减枝, 返回频繁项的list
        cand_ck_set = []
        for i in range(len(ck_1)):  # 合并两个k-1频繁项集
            cand_ck = ck_1[i]
            for j in range(i + 1, len(ck_1)):
                if ck_1[i][:ind - 1] == ck_1[j][:ind - 1]:  # 若前k-2项相同则合并
                    cand_ck.append(ck_1[j][-1])  # 合并形成频繁k项
                    if self.prune(cand_ck, ck_1):  # 检查其他k-1项集是否为频繁项集,进而减枝
                        cand_ck_set.append(cand_ck.copy())
                    cand_ck.pop()
        return cand_ck_set

    def prune(self, cand_ck_item, ck_1):
        # 根据k-1频繁项集来对k频繁项是否频繁
        for item in cand_ck_item[:-2]:
            sub_item = cand_ck_item.copy()
            sub_item.remove(item)
            if sub_item not in ck_1:
                return False
        return True

    def cal_sup(self, item):
        # 支持度计数
        s = set(item)
        sup = 0
        for t in self.data:
            if s.issubset(t):
                sup += 1
        return sup / self.N

    def cal_conf(self, sxy, X):
        # 计算置信度, sxy为产生规则的频繁项集的支持度， X为规则前件
        return sxy / self.fre_dict[tuple(X)]

    def gen_rules(self):
        # 从频繁项集中提取规则
        for i in range(1, len(self.fre_list)):
            for ind, itemset in enumerate(self.fre_list[i]):
                cand_rules = []  # 由该频繁项集产生的规则的list, 记录规则前件
                sxy = self.sup_list[i][ind]
                for item in itemset:  # 初始化后件为1个项的规则
                    X = itemset.copy()
                    X.remove(item)
                    cand_rules.append(X)

                while len(cand_rules) > 0:
                    itemset_rules = []
                    for X in cand_rules:
                        conf = self.cal_conf(sxy, X)
                        if conf > self.minconf:
                            itemset_rules.append(X)
                            Y = list(set(itemset) - set(X))
                            Y = sorted(Y)
                            self.rules_dict[(tuple(X), tuple(Y))] = conf
                    cand_rules = self.apriori_rules(itemset_rules)
        return

    def apriori_rules(self, itemset_rules):
        # 根据先验原理产生候选规则
        cand_rules = []
        for i in range(len(itemset_rules)):
            for j in range(i + 1, len(itemset_rules)):
                X = list(set(itemset_rules[i]) & set(itemset_rules[j]))  # 合并生成新的规则前件
                X = sorted(X)
                if X in cand_rules or len(X) < 1:  # 若该规则前件已经产生或者为空则跳过
                    continue
                cand_rules.append(X)
        return cand_rules

    def fit(self, data):
        self.init_param(data)
        self.apriori_fre_itemset()
        self.gen_rules()
        return


if __name__ == '__main__':
    data = [['l1', 'l2', 'l5'], ['l2', 'l4'], ['l2', 'l3'],
            ['l1', 'l2', 'l4'], ['l1', 'l3'], ['l2', 'l3'],
            ['l1', 'l3'], ['l1', 'l2', 'l3', 'l5'], ['l1', 'l2', 'l3']]

    AP = Apriori(minsup=0.2, minconf=0.6)
    AP.fit(data)
    print(AP.rules_dict)
