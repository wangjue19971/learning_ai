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
        self.gbdt_lr.train_gbdt_lr(self.load_interactions(), self.index2id, self.id2index)

    def load_interactions(self):
        # 加载交互数据
        interactions = []
        file_path = "news_rec/data/mini_news.dat"
        lines = open(file_path, encoding="utf8").readlines()
        for line in lines:
            line = line.strip().split("\t")
            user_id = line[1]
            item_id = line[2]
            label = int(line[3])
            interactions.append((user_id, item_id, label))
        return interactions

    def recall(self, user_id, user_emb_dic, item_emb_dic):
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
        if user_id in user_emb_dic:
            v = user_emb_dic[user_id]
            n_points = tree.data.shape[0]  # 获取树中点的数量
            num_neighbors = min(20, n_points)  # 确保k不超过点的数量
            dist, ind = tree.query(v.reshape(1, -1), k=num_neighbors)
            recall_results[user_id] = [item_id[i] for i in ind[0]]
        return recall_results

    def sort(self, user_id, recall_result):
        result = {}
        for item_id in recall_result[user_id]:
            score = self.gbdt_lr.predict(user_id, item_id, self.id2index)
            result[item_id] = score
        return result
    
if __name__ == "__main__":
    ras = RecallAndSort()
    user_id = "2380244356815169921"  # 指定用户ID
    
    # 加载用户和物品嵌入向量
    user_emb_dic = {}
    item_emb_dic = {}
    with open('news_rec/data/user_emb_dic.pkl', 'rb') as user_file:
        user_emb_dic = pickle.load(user_file)
    with open('news_rec/data/item_emb_dic.pkl', 'rb') as item_file:
        item_emb_dic = pickle.load(item_file)

    # 使用加载的用户和物品嵌入向量进行召回
    recall_result = ras.recall(user_id, user_emb_dic, item_emb_dic)
    print(f"召回结果: {recall_result[user_id]}")  # 修改这里以打印指定用户的召回结果

    # 对召回结果进行排序
    sort_result = ras.sort(user_id, recall_result)
    print(f"排序结果: {sort_result}")
    
# 测试集AUC:  0.6542653950341825
# 召回结果: ['4106279991666292577', '3304821928338655444', '16529623828211313387', 
#       '10265030586183854448', '6789838657778527135', '11770742125479994390', 
#       '3054423453900014545', '4080452504006450662', '12088043601288016710', 
#       '12203214915075060334', '4920758881479676410', '1719696861020491404', 
#       '8993157435258500152', '17371650719194954318', '5019091515877081211', 
#       '2996524628436276241', '16294764181418169402', '2545858582159222152', 
#       '6300686928420967038', '1000356838604552172']

# 排序结果: {'4106279991666292577': 0.40465389731811896, 
# '3304821928338655444': 0.3818071339537923, 
# '16529623828211313387': 0.3523366214190019, 
# '10265030586183854448': 0.22466121560243568, 
# '6789838657778527135': 0.22466121560243568, 
# '11770742125479994390': 0.22466121560243568, 
# '3054423453900014545': 0.3818071339537923, 
# '4080452504006450662': 0.34369277483777927, 
# '12088043601288016710': 0.4698197813065415, 
# '12203214915075060334': 0.14969070443960095, 
# '4920758881479676410': 0.4863190265514221, 
# '1719696861020491404': 0.6139146181645714, 
# '8993157435258500152': 0.41539336002500676, 
# '17371650719194954318': 0.4855973599791859, 
# '5019091515877081211': 0.3592087762948819, 
# '2996524628436276241': 0.4887125319367385, 
# '16294764181418169402': 0.3590121516299831, 
# '2545858582159222152': 0.4945621501954395, 
# '6300686928420967038': 0.33083249573252704, 
# '1000356838604552172': 0.22807822598409688}