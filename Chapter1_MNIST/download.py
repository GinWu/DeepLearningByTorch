#coding:utf-8
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


training_data = datasets.MNIST(
    root = "MNIST_data/",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.MNIST(
    root = "MNIST_data/",
    train = False,
    download = True,
    transform = ToTensor()
)

print(len(training_data))
print(len(test_data))

print(training_data[0][0])
print(training_data[0][1])