import numpy as np 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch
from torch.optim import Adam
import os

class FC(nn.Module):
    def __init__(self, n_features, hiddens, n_classes):
        '''
        n_features: 输入特征数
        hiddens: 隐藏层神经元数
        n_classes: 输出类别数
        '''
        super(FC, self).__init__()
        layers = []
        ni = n_features
        for hidden in hiddens:
            layer = nn.Linear(in_features = ni, out_features = hidden)
            layers.append(layer)
            layers.append(nn.ReLU())
            ni = hidden    
        layer = nn.Linear(in_features = ni, out_features = n_classes)
        layers.append(layer)
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        '''
        x: 输入数据
        note: 入参数目可以是一个或多个
        '''
        return self.model(x)


if __name__ == '__main__':
    root = './output'
    if not os.path.exists(root):
        os.makedirs(root)
    # 1. 随机产生数据
    n_classes = 2
    n_features = 10
    x, y = make_classification(n_samples=1000, n_features=n_features, n_classes=n_classes, random_state=24)
    # print(f'X.shape: {x.shape}, y.shape: {y.shape}, class: {np.unique(y)}')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}')
    # print(f'x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}')
    
    # 2. 创建模型
    model = FC(n_features=n_features, hiddens=[64, 32], n_classes=n_classes)
    loss_fn = nn.CrossEntropyLoss()
    train_op = Adam(model.parameters(), lr=0.001)
    # 2.1 恢复模型
    names = os.listdir(root)
    if len(names) > 0:
        names.sort()
        path = os.path.join(root, names[-1])
        original_model = torch.load(path)
        # 2.2 模型参数恢复
        if isinstance(original_model, nn.Module):
            model.load_state_dict(original_model.state_dict())
        else:
            model.load_state_dict(original_model)
        print(f'Load model from {path}')
    
    # 3. 训练模型
    total_epochs = 100
    batch_size = 64
    total_train_samples = len(x_train)
    for epoch in range(total_epochs):
        model.train() # 设置为训练模式
        # 3.1 随机打乱数据
        total_batch = total_train_samples // batch_size
        idx = np.random.permutation(total_train_samples)
        for batch in range(total_batch):
            # 3.2 取出batch数据
            start = batch * batch_size
            end = (batch + 1) * batch_size
            # 3.3 处理最后一个batch
            if end > total_train_samples:
                end = total_train_samples
            batch_idx = idx[start:end]
            batch_x = torch.tensor(x_train[batch_idx], dtype=torch.float32)
            batch_y = torch.tensor(y_train[batch_idx], dtype=torch.long)
            
            # 3.4 前向过程，计算loss
            train_op.zero_grad() # 梯度清零
            pred = model(batch_x) # 前向计算，(64,2)
            loss = loss_fn(pred, batch_y) # 计算loss
            
            # 3.5 反向传播
            loss.backward() # 反向传播计算梯度
            train_op.step() # 更新参数
    
        # 4. 测试模型
        model.eval() # 设置为测试模式
        with torch.no_grad():
            x_test = torch.tensor(x_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.long)
            pred = model(x_test)
            pred = torch.argmax(pred, dim=1)
            acc = (pred == y_test).sum().item() / len(y_test)
            # print(f'Accuracy: {acc}')
            
        # 5. 模型保存,可以保存任意对象，因为底层使用pickle
        # torch.save(model, os.path.join(root, f'./fc_model_{epoch: 04d}.pth'))
        torch.save(model.state_dict(), os.path.join(root, f'./fc_model_{epoch: 04d}.pth'))