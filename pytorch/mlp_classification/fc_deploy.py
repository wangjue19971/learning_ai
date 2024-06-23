import torch
import torch.nn as nn
import numpy as np
import os
from fc_demo import FC

class Predictor:
    def __init__(self, root):
        n_classes = 2
        n_features = 10
        self.model = FC(n_features=10, hiddens=[64, 32], n_classes=2)
        
        root = './output'
        names = os.listdir(root)
        if len(names) > 0:
            names.sort()
            path = os.path.join(root, names[-1])
            original_model = torch.load(path)
            if isinstance(original_model, nn.Module):
                self.model.load_state_dict(original_model.state_dict())
            else:
                self.model.load_state_dict(original_model)
        else:
            raise Exception(f'No model found!:{root}')
        
        # 模型评估
        self.model.eval()
    
    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y = self.model(x)
            p = torch.nn.functional.softmax(y, dim=0)
            y,i = torch.max(p, dim=0)
            return y.numpy()

if __name__ == '__main__':
    predictor = Predictor('./output')
    x = np.random.randn(10)
    y = predictor.predict(x)
    print(f'x: {x}, y: {y}')
        