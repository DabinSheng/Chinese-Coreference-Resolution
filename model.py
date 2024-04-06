import torch
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 显存不够，跑不起来多层感知机
        # self.fc1 = nn.Linear(5205, 1000)
        # self.sigmoid=nn.Sigmoid()
        # self.fc2 = nn.Linear(1000, 500)
        # self.relu=nn.ReLU()
        # self.dropout=nn.Dropout(0.5)
        # self.fc3=nn.Linear(500,1)
        # self.output=nn.Sigmoid()
        self.fc1 = nn.Linear(5205, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x=x.view(-1,5205)
        x = self.fc1(x)
        x=self.dropout(x)
        x=self.sigmoid(x)
        # x=self.fc2(x)
        # x=self.relu(x)
        # x = self.dropout(x)
        # x=self.fc3(x)
        # x=self.output(x)
        return x