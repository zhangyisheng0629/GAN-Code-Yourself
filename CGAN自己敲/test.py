#!/usr/bin/python
# author eson
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NUM_CLASSES = 1000
GEN_EMBEDDING = 100
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10


transforms = transforms.Compose(
    [
        #Resize的第一个参数如果是int，如果H>W，则会按照[...,H/W*size,size]切割图片
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])

    ]
)
dataset = datasets.CIFAR10(root='../datasets/', train=True, transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

targrts=np.array(dataset.targets)
idx=np.where(targrts==6)

idx=np.array(idx)
print(idx.shape)
a=np.array([1,2,3])
print(a)

