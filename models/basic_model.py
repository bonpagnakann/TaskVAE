import random

import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor
from torch import from_numpy, ones, zeros
from torch.utils import data
from . import modified_linear

PATH_TO_SAVE_WEIGHTS = 'saved_weights/'

class Net(nn.Module):
    def __init__(self, n_classes, icarl=False, cosine_liner=False):
        super(Net, self).__init__()
        self.icarl = icarl

        self.conv1 = nn.Conv1d(6, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm1d(64)

        self.fc0 = nn.Linear(192,64)
        self.fc1 = nn.Linear(64, 32)
        final_dim = self.fc1.out_features
        self.fc = modified_linear.CosineLinear(final_dim, n_classes) if cosine_liner \
            else nn.Linear(final_dim, n_classes)

    def forward(self, x, is_feature_input=False):
        
        if is_feature_input:
            # Input directly passed to the fully connected layers
            x = self.fc1(x)  
            x = F.relu(x)
            return self.fc(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc0(x))
        x = self.fc1(x)
        if self.icarl == True:
            x = F.normalize(x, p=2, dim=1)
        else:
            x = F.relu(x)

        x = self.fc(x)

        return x

    def feature_extraction(self, x, fetril=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = x.reshape(x.shape[0], -1)
        if fetril == True:
            x = F.relu(self.fc0(x))
        else:
            x = F.relu(self.fc0(x))
            x = self.fc1(x)
            if self.icarl == True:
                x = F.normalize(x, p=2, dim=1)

        return x


class Net2(nn.Module):
    def __init__(self, dim):
        super(Net2, self).__init__()

        self.conv1 = nn.Conv1d(6, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm1d(64)

        self.fc0 = nn.Linear(192,dim)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc0(x))

        return x


class Dataset(data.Dataset):
    def __init__(self, features, labels):
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        X = from_numpy(self.features[idx])
        y = self.labels[idx]
        y = LongTensor([y])
        return X, y

    def get_sample(self, sample_size):
        return random.sample(self.features, sample_size)
