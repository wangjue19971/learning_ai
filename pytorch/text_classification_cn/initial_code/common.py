import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def get_vocab_mapping(vocab_path): # 词表路径, 返回词表字典
    '''
    vocab_path: 词表路径
    return: 词表字典，key为单词，value为单词id
    '''
    vocab_path = 'text_classification_cn/data.csv'
    df = pd.read_csv(vocab_path)
    vocab = {
        'UNK': 0, # 未知单词
        'PAD': 1, # 填充单词
        'NUM': 2, # 数字单词
        'POT': 3, # 标点符号
    }
    index = 4
    for text in df['text']:
        for word in text:
            if word in vocab:
                continue # 单词已经存在, 跳过
            vocab[word] = index
            index += 1
    return vocab


class Dataset_cn(Dataset):
    def __init__(self, word2id):
        super(Dataset_cn, self).__init__()
        data_path = 'text_classification_cn/data.csv'
        self.word2id = word2id
        self.xs, self.ys, self.lengths = self.get_data(data_path)
    
    def split_text(self, text):
        '''
        针对中文文本进行分词
        text: 文本字符串
        return: 单词id列表
        '''
        ids = []
        for word in text: 
            word = word.strip() # 去除单词前后空格
            if len(word) == 0:
                continue
            # 1. 判断是否为数字
            if word in '0123456789':
                ids.append(self.word2id['NUM'])
                continue
            # 2. 判断是否为标点符号
            if word in ',.!?':
                ids.append(self.word2id['POT'])
                continue
            # 3. 判断是否在词表中, 不在则使用UNK
            if word in self.word2id:
                ids.append(self.word2id[word])
            else:
                ids.append(self.word2id['UNK'])
        return ids
       
    def get_data(self, data_path):
        df = pd.read_csv(data_path)
        df = df[['text', 'label']]
        xs = []
        ys = []
        lengths = []
        for value in df.values:
            text, label = value
            x = self.split_text(text)
            y = int(label)
            xs.append(x)
            ys.append(y)
            lengths.append(len(x))
        return xs, ys, lengths

    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, index):
        x = self.xs[index]
        y = self.ys[index]
        length = self.lengths[index]
        return x, y, length

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.n_classes = 2 # 类别数
        self.vocab_size = 100 # 词表大小
        self.embedding_size = 128 # 词向量维度
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_size) # 词向量层
        self.classify_layer = nn.Sequential(
            nn.Linear(self.embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, self.n_classes)
        )        
        
    def forward(self, x, lengths):
        '''
        note: n表示每个批次的样本数量，t表示每个批次的时间步长；在不同批次中t大小不同，但n相同
        x: 特征属性，形状为(n,t),全部为单词id
        lengths: 样本实际长度信息，形状为(n,)
        return: 模型输出，形状为(n,2)，为各个类别的置信度
        '''
        # 1. 将每个时刻的单词id转换成词向量，形状为(n,t,embedding_size)
        x = self.embedding_layer(x)
        
        # 2. 将每个样本t时刻的词向量合并成为一个向量, 形状为(n,embedding_size)
        n, t, embedding_size  = x.shape
        mask = torch.zeros((n, t)).to(torch.long)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1 # 有效位置为1
        mask = mask[..., None].expand_as(x) # (n,t,embedding_size), 1表示有效，0表示无效
        x = x * mask # 无效位置的词向量为0
        x = torch.sum(x, dim=1)/ lengths.to(x.dtype).view(-1, 1) # 求平均, (n,embedding_size)
        
        #3. 全连接层分类
        output = self.classify_layer(x)
        return output
    
    
if __name__ == '__main__':
    w = get_vocab_mapping('text_classification_cn/data.csv')
    dataset = Dataset_cn(w)
    print(len(dataset))
    print(dataset[6])


