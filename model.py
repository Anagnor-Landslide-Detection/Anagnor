import torch
import torch.nn as nn
import torch.nn.functional as F


class AnagnorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 12, 3)
        self.pool = nn.MaxPool2d(10, 10)
        self.conv2 = nn.Conv2d(12, 3, 3)
        self.fc1 = nn.Linear(300, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.dropout = nn.Dropout(0.3)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sig(x)
        return x

if __name__ == "__main__":
    print("Hello")
    net = AnagnorModel()
    p = net(torch.randn(4,32,1024,1024))

    print(p)