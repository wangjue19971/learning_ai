import matrixcf
import numpy as np
from sklearn.neighbors import BallTree
import pickle

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

        # 加载矩阵分解模型
        self.mcf = matrixcf.MCF()
        self.mcf.load_model()

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

    def sort(self, user, recall_result):
        # 修改排序逻辑以接受召回结果的用户ID列表
        user_indices = [self.id2index[user]] * len(recall_result[user])  # 对于每个召回的物品，用户索引是相同的
        item_indices = [self.id2index[item] for item in recall_result[user]]  # 将召回的物品ID转换为索引

        # 确保输入格式正确：将列表转换为NumPy数组，因为大多数深度学习框架都支持NumPy数组作为输入
        user_indices_np = np.array(user_indices).reshape(-1, 1)  # 转换为NumPy数组并确保形状正确
        item_indices_np = np.array(item_indices).reshape(-1, 1)
        
        # 使用模型进行预测
        res = self.mcf.infer([user_indices_np, item_indices_np])  # 修复这里的调用，确保传递正确的参数
        res = res['pred'].numpy().tolist()
        result = {}
        # 将预测结果转换为字典形式
        for i, item in enumerate(recall_result[user]):
            result[item] = res[i]
        return result
    
# 加载用户和物品嵌入向量
user_emb_dic = {}
item_emb_dic = {}

# 从文件加载用户和物品嵌入向量
with open('user_emb_dic.pkl', 'rb') as user_file:
    user_emb_dic = pickle.load(user_file)
with open('item_emb_dic.pkl', 'rb') as item_file:
    item_emb_dic = pickle.load(item_file)

if __name__ == "__main__":
    ras = RecallAndSort()
    user_id = "2380244356815169921"  # 指定用户ID
    # 使用加载的用户和物品嵌入向量进行召回
    recall_result = ras.recall(user_id, user_emb_dic, item_emb_dic)
    print(f"召回结果: {recall_result[user_id]}")  # 修改这里以打印指定用户的召回结果
    # 对召回结果进行排序
    sort_result = ras.sort(user_id, recall_result)
    print(f"排序结果: {sort_result}")
    
# 召回结果: ['4106279991666292577', '3304821928338655444', 
# '16529623828211313387', '10265030586183854448', '6789838657778527135',
# '11770742125479994390', '3054423453900014545', '4080452504006450662', 
# '12088043601288016710', '12203214915075060334', '4920758881479676410', 
# '1719696861020491404', '8993157435258500152', '17371650719194954318', 
# '5019091515877081211', '2996524628436276241', '16294764181418169402', 
# '2545858582159222152', '6300686928420967038', '1000356838604552172']

# 排序结果: {'4106279991666292577': [1.0517382621765137], 
# '3304821928338655444': [0.6403272151947021], 
# '16529623828211313387': [0.6402989029884338], 
# '10265030586183854448': [0.6204764246940613], 
# '6789838657778527135': [0.6371762156486511], 
# '11770742125479994390': [0.5666526556015015], 
# '3054423453900014545': [0.5370762944221497], 
# '4080452504006450662': [0.9547314643859863], 
# '12088043601288016710': [0.8341240286827087], 
# '12203214915075060334': [0.9490894079208374], 
# '4920758881479676410': [0.6405525207519531], 
# '1719696861020491404': [0.4412810802459717], 
# '8993157435258500152': [0.23086684942245483], 
# '17371650719194954318': [0.5923388004302979], 
# '5019091515877081211': [0.9422974586486816], 
# '2996524628436276241': [1.2333073616027832], 
# '16294764181418169402': [0.29258617758750916], 
# '2545858582159222152': [0.2570118308067322], 
# '6300686928420967038': [0.5597556829452515], 
# '1000356838604552172': [0.6842941045761108]}