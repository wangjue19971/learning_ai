import numpy as np
import pickle
from sklearn.neighbors import BallTree
from GBDT_LR import GBDT_LR

class RecallAndSort:
    def __init__(self):
        # 初始化索引到ID和ID到索引的映射字典
        self.index2id = {}
        self.id2index = {}
        # 从文件中读取索引和ID的映射关系
        with open("news_rec/data/index") as lines:
            for line in lines:
                line = line.strip().split("\t")
                self.index2id[int(line[1])] = line[0]
                self.id2index[line[0]] = int(line[1])

        # 加载GBDT+LR模型
        self.gbdt_lr = GBDT_LR()
        self.gbdt_lr.train_gbdt_lr(self.load_interactions())

    def load_interactions(self):
        # 打开文件
        file_path = "news_rec/data/mini_news.dat"
        lines = open(file_path, encoding="utf8").readlines()

        # 初始化一个空列表来存储元组
        interactions = []

        # 遍历每一行，提取user, item, ctr，并将它们组合成元组
        for line in lines:
            line = line.strip().split("\t")
            user_id = int(line[1])
            item_id = int(line[2])
            label = int(line[3])
            interactions.append((user_id, item_id, label))

        return interactions

    def recall(self, user_id, item_emb_dic):
        # 召回逻辑
        item_id = []
        item_emb_numpy = []
        # 将项目ID和嵌入向量转换为列表形式
        for k, v in item_emb_dic.items():
            item_id.append(k)
            item_emb_numpy.append(v)
        # 使用BallTree构建物品的空间索引，以便快速查询
        tree = BallTree(np.squeeze(np.array(item_emb_numpy)), leaf_size=10)
        recall_results = {}
        # 对指定用户嵌入向量进行查询，找到最近的k个物品
        if user_id in item_emb_dic:
            v = item_emb_dic[user_id]
            n_points = tree.data.shape[0]  # 获取树中点的数量
            num_neighbors = min(20, n_points)  # 确保k不超过点的数量
            dist, ind = tree.query(v.reshape(1, -1), k=num_neighbors)
            recall_results[user_id] = [item_id[i] for i in ind[0]]
        return recall_results

    def sort(self, user_id, recall_result):
        # 对召回结果进行排序
        sorted_result = {}
        for item in recall_result[user_id]:
            sorted_result[item] = self.gbdt_lr.predict(user_id, item)

        # 根据预测分数进行排序
        sorted_result = dict(sorted(sorted_result.items(), key=lambda x: x[1], reverse=True))
        return sorted_result

if __name__ == "__main__":
    ras = RecallAndSort()

    # 加载用户和物品嵌入向量
    with open('news_rec/data/user_emb_dic.pkl', 'rb') as user_file:
        user_emb_dic = pickle.load(user_file)
    with open('news_rec/data/item_emb_dic.pkl', 'rb') as item_file:
        item_emb_dic = pickle.load(item_file)

    user_id = '2380244356815169921'  # 指定用户ID
    # 使用加载的用户和物品嵌入向量进行召回
    recall_result = ras.recall(user_id, item_emb_dic)
    print(f"召回结果: {recall_result[user_id]}")  # 修改这里以打印指定用户的召回结果
    # 对召回结果进行排序
    sort_result = ras.sort(user_id, recall_result)
    print(f"排序结果: {sort_result}")