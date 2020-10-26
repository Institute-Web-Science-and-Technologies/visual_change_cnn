# Copyright (C) 2020 Daniel Vossen
# see COPYING for further details

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(6, 48, 5)
        self.conv2 = nn.Conv2d(48, 96, 5)
        self.conv3 = nn.Conv2d(96, 192, 3)
        self.conv4 = nn.Conv2d(192, 384, 3)
        self.conv5 = nn.Conv2d(384, 512, 3)
        self.conv6 = nn.Conv2d(512, 512, 3)
        self.conv7 = nn.Conv2d(512, 512, 3)
        self.conv8 = nn.Conv2d(512, 512, 3)
        self.fc1 = nn.Linear(49152, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = F.relu(self.conv7(x))
        x = self.pool(F.relu(self.conv8(x)))
        nfs = self.num_flat_features(x)
        x = x.view(-1, nfs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


