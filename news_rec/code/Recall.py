import numpy as np
from sklearn.neighbors import BallTree
import pickle
from MCF import MCF

class Recall:
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

        # 加载矩阵分解模型
        self.mcf = MCF()
        self.mcf.load_model()

    def recall(self, user_id, user_emb_dic, item_emb_dic):
        # 召回逻辑
        item_id = []
        item_emb_numpy = []
        for k, v in item_emb_dic.items():
            item_id.append(k)
            item_emb_numpy.append(v)
        # 使用BallTree构建物品的空间索引，以便快速查询
        tree = BallTree(np.squeeze(np.array(item_emb_numpy)), leaf_size=10)
        recall_results = {}
        # 对指定用户嵌入向量进行查询，找到最近的k个物品
        if user_id in user_emb_dic:
            v = user_emb_dic[user_id]
            n_points = tree.data.shape[0]
            num_neighbors = min(100, n_points)
            dist, ind = tree.query(v.reshape(1, -1), k=num_neighbors)
            recall_results[user_id] = [item_id[i] for i in ind[0]]
        return recall_results

if __name__ == "__main__":
    # 加载用户和物品嵌入向量
    with open('news_rec/data/user_emb_dic.pkl', 'rb') as user_file:
        user_emb_dic = pickle.load(user_file)
    with open('news_rec/data/item_emb_dic.pkl', 'rb') as item_file:
        item_emb_dic = pickle.load(item_file)

    recall = Recall()
    user_id = "2380244356815169921"
    recall_result = recall.recall(user_id, user_emb_dic, item_emb_dic)
    print(f"召回结果: {recall_result[user_id]}")