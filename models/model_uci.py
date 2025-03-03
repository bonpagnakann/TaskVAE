import torch.nn as nn
import torch.nn.functional as F

class cnNet(nn.Module):
    def __init__(self, data_size, n_classes):
        super(cnNet, self).__init__()
        self.conv1 = nn.Conv1d(6, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=5, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc0 = nn.Linear(1008,100)
        self.fc5 = nn.Linear(100, n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        #print("Shape after conv1:", x.shape)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = x.reshape(batch_size, -1)
        x = self.fc0(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x
