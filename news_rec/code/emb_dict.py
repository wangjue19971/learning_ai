import matrixcf
import numpy as np
import pickle

# load index2id
index2id = {}
with open("news_rec/data/index") as lines:
    for line in lines:
        line = line.strip().split("\t")
        index2id[int(line[1])] = line[0]

# load model
mcf = matrixcf.MCF()
mcf.load_model()
test_ds = mcf.init_dataset("news_rec/data/test", is_train=False)
user_emb_dic = {}
item_emb_dic = {}

for ds in test_ds:
    ds.pop("ctr")
    user_index = ds['user'].numpy()
    item_index = ds['item'].numpy()
    res = mcf.infer(ds)
    user_emb = res['user_emb'].numpy()
    item_emb = res['item_emb'].numpy()

    for k, v in zip(user_index, user_emb):
        user_emb_dic[index2id[k]] = v
    for k, v in zip(item_index, item_emb):
        item_emb_dic[index2id[k]] = v

print(user_emb_dic)

# 保存用户嵌入向量字典到文件
with open('news_rec/data/user_emb_dic.pkl', 'wb') as user_file:
    pickle.dump(user_emb_dic, user_file)

# 保存物品嵌入向量字典到文件
with open('news_rec/data/item_emb_dic.pkl', 'wb') as item_file:
    pickle.dump(item_emb_dic, item_file)