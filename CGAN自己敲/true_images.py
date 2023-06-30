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
from Samplers import TrueSampler

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
        # Resize的第一个参数如果是int，如果H>W，则会按照[...,H/W*size,size]切割图片
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])

    ]
)

# dataset = datasets.MNIST(root="../datasets/", train=True, transform=transforms,
#                          download=True)
# dataset=datasets.ImageFolder(root='/users/uestc1/zys/Datasets/ILSVRC2012_img_train',transform=transforms,)
dataset = datasets.CIFAR10(root='../datasets/', train=True, transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
writer_real_per_cls = SummaryWriter(f'logs/real_per_cls')

true_clses = [0, 1, 2, 3, 4, 5, 6]
for cls in true_clses:
    imgs = np.array([sample.transpose(2, 0, 1)
                     for sample, target in zip(dataset.data, dataset.targets) if target == cls][:32])
    imgs = imgs.astype('float32')
    imgs = torch.tensor(imgs, )
    imgs = transforms(imgs)
    imgs_grid = torchvision.utils.make_grid(imgs, normalize=True)
    writer_real_per_cls.add_image(f'Real_per_cls {cls}', imgs_grid)


class Sampler:
    def __init__(self, num_idx, num_per_cls):
        pass

    def write(self, writer, image_grid):
        pass

# imgs_per_cls=
# writer_real_per_cls.add_image(f'Real_Per_Cls {}')
