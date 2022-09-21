import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Convolution(nn.Module):
    def __init__(self):
        super(Convolution, self).__init__()
        self.flatten = nn.Flatten()
        # TODO conv2d, MaxPool2d 参数
        self.conv1 = nn.Conv2d(1, 32, stride = 1, kernel_size = 5, padding = "same")
        self.conv2 = nn.Conv2d(32, 64, stride = 1, kernel_size = 5, padding = "same")
        self.maxPool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.linear1 = nn.Linear(64 * 7 * 7, 1024)
        self.linear2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout()

    def forward(self, x):
        h_conv1 = self.relu(self.conv1(x))
        h_pool1 = self.maxPool(h_conv1)
        h_conv2 = self.relu(self.conv2(h_pool1))
        
        h_pool2 = self.maxPool(h_conv2)
        h_pool2_flat = self.flatten(h_pool2)
        
        h_fc1 = self.relu(self.linear1(h_pool2_flat))
        h_fc1_drop = self.drop_out(h_fc1)
        y_conv = self.linear2(h_fc1_drop)

        return y_conv
        