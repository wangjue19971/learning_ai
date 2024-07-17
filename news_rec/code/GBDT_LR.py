import numpy as np
import lightgbm as lgb
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class GBDT_LR:
    def __init__(self):
        self.gbdt_model = None
        self.lr_model = None

    def train_gbdt_lr(self, interactions, index2id, id2index):
        # 生成训练数据
        X = []
        y = []
        for (user_id, item_id, label) in interactions:
            if user_id in id2index and item_id in id2index:
                user_index = id2index[user_id]
                item_index = id2index[item_id]
                feature = [user_index, item_index]
                X.append(feature)
                y.append(label)
        
        X = np.array(X)
        y = np.array(y)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 训练GBDT模型
        self.gbdt_model = lgb.LGBMClassifier()
        self.gbdt_model.fit(X_train, y_train)

        # 生成GBDT模型的叶节点索引特征
        gbdt_train_features = self.gbdt_model.predict(X_train, pred_leaf=True)
        gbdt_test_features = self.gbdt_model.predict(X_test, pred_leaf=True)

        # 训练LR模型
        self.lr_model = LogisticRegression(max_iter=1000)
        self.lr_model.fit(gbdt_train_features, y_train)

        # 测试LR模型
        lr_pred = self.lr_model.predict_proba(gbdt_test_features)[:, 1]
        print("测试集AUC: ", roc_auc_score(y_test, lr_pred))

    def predict(self, user_id, item_id, id2index):
        # 确保GBDT模型已经被训练
        if self.gbdt_model is None:
            print("GBDT模型尚未训练，请先调用train_gbdt_lr方法进行训练。")
            return 0.0

        if user_id in id2index and item_id in id2index:
            user_index = id2index[user_id]
            item_index = id2index[item_id]
            feature = np.array([[user_index, item_index]])
            gbdt_feature = self.gbdt_model.predict(feature, pred_leaf=True)
            lr_pred = self.lr_model.predict_proba(gbdt_feature)[:, 1]
            return lr_pred[0]
        return 0.0

if __name__ == "__main__":
    gbdt_lr = GBDT_LR()
    
    # 加载索引和ID的映射关系
    index2id = {}
    id2index = {}
    with open("news_rec/data/index") as lines:
        for line in lines:
            line = line.strip().split("\t")
            index2id[int(line[1])] = line[0]
            id2index[line[0]] = int(line[1])

    # 打开文件
    file_path = "news_rec/data/mini_news.dat"
    lines = open(file_path, encoding="utf8").readlines()

    # 初始化一个空列表来存储元组
    interactions = []

    # 遍历每一行，提取user, item, ctr，并将它们组合成元组
    for line in lines:
        line = line.strip().split("\t")
        user_id = line[1]
        item_id = line[2]
        label = int(line[3])
        interactions.append((user_id, item_id, label))

    gbdt_lr.train_gbdt_lr(interactions, index2id, id2index)