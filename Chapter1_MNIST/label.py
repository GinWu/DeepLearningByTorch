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

labels_map = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

for i in range(20):
    sample_idx = torch.randint(len(training_data), size = (1,)).item()
    img, label = training_data[sample_idx]
    print("mnist_train_%d.jpg label: %s" % (sample_idx, label))