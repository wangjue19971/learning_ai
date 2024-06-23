import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 定义一些常量
UNKNOWN_WORD = 'UNK'
PADDING_WORD = 'PAD'
NUMBER_WORD = 'NUM'
PUNCTUATION_WORD = 'POT'

def load_data(data_path):
    """加载数据"""
    return pd.read_csv(data_path)

def build_vocab(df, start_index=4):
    """构建词表"""
    vocab = {
        UNKNOWN_WORD: 1,
        PADDING_WORD: 0,
        NUMBER_WORD: 2,
        PUNCTUATION_WORD: 3,
    }
    for text in df['text']:
        for word in text:
            if word not in vocab:
                vocab[word] = start_index
                start_index += 1
    return vocab

class TextClassificationDataset(Dataset):
    """文本分类数据集"""
    def __init__(self, df, word2id):
        super().__init__()
        self.pad_id = word2id[PADDING_WORD]
        self.word2id = word2id # self代表可以在该类的其他方法中使用该属性
        # df = load_data(data_path) # 没有self，只能在__init__方法中使用，只是创建实例时调用一次
        self.xs, self.ys, self.lengths = self.process_data(df)

    def process_data(self, df):
        """处理数据"""
        xs, ys, lengths = [], [], []
        for text, label in df[['text', 'label']].values:
            x = self.text_to_ids(text)
            y = int(label)
            xs.append(x)
            ys.append(y)
            lengths.append(len(x))
        return xs, ys, lengths

    def text_to_ids(self, text):
        """将文本转换为id列表"""
        ids = []
        for word in text:
            word = word.strip()
            if not word:
                continue
            if word.isdigit():
                ids.append(self.word2id[NUMBER_WORD])
            elif word in ',.!?':
                ids.append(self.word2id[PUNCTUATION_WORD])
            else:
                ids.append(self.word2id.get(word, self.word2id[UNKNOWN_WORD]))
        return ids

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        return self.xs[index], self.ys[index], self.lengths[index]
    
    def collate_fn(self, batch): # batch是一个列表，列表中的每个元素是__getitem__方法的返回值
        '''
        batch: [(x1, y1, l1), (x2, y2, l2), ..., (xn, yn, ln)]
        return: xs, ys, lengths
        '''
        xs, ys, lengths = [], [], []
        max_length = max([l for _, _, l in batch]) 
        for x, y, l in batch:
            ys.append(y)
            lengths.append(l)
            x = x + [self.pad_id] * (max_length - l) # 补齐, 使每个样本的长度相同
            xs.append(x)
        return torch.tensor(xs), torch.tensor(ys), torch.tensor(lengths)
              

class SimpleNetwork(nn.Module):
    """简单的神经网络"""
    def __init__(self, vocab_size, embedding_size=128, n_classes=2):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.classify_layer = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x, lengths):
        x = self.embedding_layer(x)
        mask = self.create_mask(x, lengths)
        x = x * mask
        x = torch.sum(x, dim=1) / lengths.view(-1, 1).to(x.dtype)
        return self.classify_layer(x)

    def create_mask(self, x, lengths):
        """创建掩码"""
        n, t = x.shape[:2]
        mask = torch.zeros((n, t)).to(torch.long)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1
        return mask.unsqueeze(-1)


def calculate_accuracy(y_pred, y_true):
    """计算准确率"""
    y_pred = torch.argmax(y_pred, dim=1)
    correct = (y_pred == y_true).float()
    accuracy = correct.sum() / len(correct)
    return accuracy
