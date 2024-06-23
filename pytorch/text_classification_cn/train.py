import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os

import sys
sys.path.append("/Users/wangjue/algo/Learning_RecSys/pytorch")
from text_classification_cn import _common

# 加载数据
def load_data(df):
    # 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # 构建词汇表
    word2id = _common.build_vocab(train_df)
    # 创建训练集和测试集
    train_dataset = _common.TextClassificationDataset(train_df, word2id)
    test_dataset = _common.TextClassificationDataset(test_df, word2id)
    # 创建数据加载器
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                             collate_fn=train_dataset.collate_fn)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                            collate_fn=test_dataset.collate_fn)
    return trainloader, testloader, word2id

# 训练和评估模型
def train_and_evaluate_model(model, trainloader, testloader, epochs, loss_fn, optimizer, writer):
    best_acc = 0.0
    best_loss = 0.0
    best_model_state = None
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        # 训练模型
        for i, (inputs, labels, lengths) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += _common.calculate_accuracy(outputs, labels).item()
        avg_loss = total_loss / len(trainloader)
        avg_acc = total_acc / len(trainloader)
        writer.add_scalar('Loss', avg_loss, epoch)
        writer.add_scalar('Accuracy', avg_acc, epoch)
        print(f'Training, Epoch: {epoch}, Loss: {avg_loss}, Accuracy: {avg_acc}')
        # 评估模型
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_acc = 0
            for i, (inputs, labels, lengths) in enumerate(testloader):
                outputs = model(inputs, lengths)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                total_acc += _common.calculate_accuracy(outputs, labels).item()
            avg_loss = total_loss / len(testloader)
            avg_acc = total_acc / len(testloader)
            writer.add_scalar('Validation/Loss', avg_loss, epoch)
            writer.add_scalar('Validation/Accuracy', avg_acc, epoch)
            print(f'Validation, Epoch: {epoch}, Loss: {avg_loss}, Accuracy: {avg_acc}')
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_loss = avg_loss
                best_model_state = model.state_dict()
    return best_model_state, best_acc, best_loss

# 保存模型
def save_model(model_state, word2id, best_acc, best_loss, model_dir):
    torch.save({
                'model_state_dict': model_state,
                'word2id': word2id,
                'best_acc': best_acc,
                'best_loss': best_loss
                }, os.path.join(model_dir, f'best_model.pth'))
    print(f'Best model, Loss: {best_loss}, Accuracy: {best_acc}')

# 转换模型为jit格式
def convert_model_to_jit(model, input_example, lengths_example):
    traced_model = torch.jit.trace(model, (input_example, lengths_example))
    traced_model.save("text_classification_cn/output/traced_model.pt")
    print("Model has been converted to JIT format and saved to traced_model.pt")

# 主函数
def training():
    # 加载数据
    df = _common.load_data('text_classification_cn/data.csv')
    output_dir = 'text_classification_cn/output'
    model_dir = os.path.join(output_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    lr = 0.01
    epochs = 10
    original_model = None
    names = os.listdir(model_dir)
    if len(names) > 0:
        names.sort()
        path = os.path.join(model_dir, names[-1]) 
        original_model = torch.load(path)
        print(f'Load model from {path}')
    if original_model is None:
        word2id = _common.build_vocab(df)
    else:
        word2id = original_model['word2id']
    # 加载数据
    trainloader, testloader, word2id = load_data(df)
    writer = SummaryWriter('text_classification_cn/logs')
    # 创建模型
    model = _common.SimpleNetwork(vocab_size=len(word2id))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 训练和评估模型
    best_model_state, best_acc, best_loss = train_and_evaluate_model(model, trainloader, testloader, epochs, loss_fn, optimizer, writer)
    # 保存模型
    save_model(best_model_state, word2id, best_acc, best_loss, model_dir)
    # 转换模型为jit格式
    input_example = torch.rand(1, 10).long()  # 这个例子假设输入是1x10的张量，你需要根据你的模型输入进行修改
    lengths_example = torch.tensor([10]).long()  # 这个例子假设长度是10，你需要根据你的模型输入进行修改
    convert_model_to_jit(model, input_example, lengths_example)
    writer.close()


if __name__ == '__main__':
    training()
    
# Best model, Loss: 0.17910093645731065, Accuracy: 0.9356211949156844