import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class GBDT_LR:
    def __init__(self):
        self.gbdt_model = None
        self.lr_model = None

    def train_gbdt_lr(self, interactions):
        # 生成训练数据
        X = []
        y = []
        for (user_id, item_id, label) in interactions:
            # 将ID值缩放到较小的范围内
            feature = np.array([user_id % 10000, item_id % 10000], dtype=np.int32)
            X.append(feature)
            y.append(int(label))

        X = np.array(X)
        y = np.array(y)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 训练GBDT模型
        train_data = lgb.Dataset(X_train, label=y_train)
        self.gbdt_model = lgb.train({}, train_data)

        # 生成GBDT模型的叶节点索引特征
        gbdt_train_features = self.gbdt_model.predict(X_train, pred_leaf=True)
        gbdt_test_features = self.gbdt_model.predict(X_test, pred_leaf=True)

        # 将叶节点索引特征展平为2D数组
        gbdt_train_features = gbdt_train_features.reshape(gbdt_train_features.shape[0], -1)
        gbdt_test_features = gbdt_test_features.reshape(gbdt_test_features.shape[0], -1)

        # 训练LR模型
        self.lr_model = LogisticRegression()
        self.lr_model.fit(gbdt_train_features, y_train)

        # 测试LR模型
        lr_pred = self.lr_model.predict_proba(gbdt_test_features)[:, 1]
        print("测试集AUC: ", roc_auc_score(y_test, lr_pred))

    def predict(self, user_id, item_id):
        # 确保GBDT模型已经被训练
        if self.gbdt_model is None:
            print("GBDT模型尚未训练，请先调用train_gbdt_lr方法进行训练。")
            return 0.0

        feature = np.array([user_id % 10000, item_id % 10000], dtype=np.int32).reshape(1, -1)
        gbdt_feature = self.gbdt_model.predict(feature, pred_leaf=True).reshape(1, -1)
        lr_pred = self.lr_model.predict_proba(gbdt_feature)[:, 1]
        return lr_pred[0]

if __name__ == "__main__":
    gbdt_lr = GBDT_LR()

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
        label = line[3]
        interactions.append((user_id, item_id, label))

    gbdt_lr.train_gbdt_lr(interactions)