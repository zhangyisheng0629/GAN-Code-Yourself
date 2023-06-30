#!/usr/bin/python
# author eson
import os

import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator, Generator, initial
from utils import gradient_penalty
from Samplers import FakeSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 3e-5
BATCH_SIZE = 64
IMAGE_SIZE = 32
CHANNELS_IMG = 3
NUM_CLASSES = 10
GEN_EMBEDDING = 100
Z_DIM = 100
NUM_EPOCHS = 50
FEATURES_DISC = 32
FEATURES_GEN = 32
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

transforms = transforms.Compose(
    [
        # Resize的第一个参数如果是int，如果H>W，则会按照[...,H/W*size,size]切割图片
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])

    ]
)

# dataset = datasets.MNIST(root="../datasets/", train=True, transform=transforms,
#                          download=True)
# imagenet
# dataset = datasets.ImageFolder(root='/users/uestc1/zys/Datasets/ILSVRC2012_img_train', transform=transforms, )
dataset = datasets.CIFAR10(root='../datasets/', train=True, transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(
    Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, IMAGE_SIZE).to(device)

if os.path.exists('model/gen') and os.path.exists('model/gen'):
    gen.load_state_dict(torch.load('model/gen'))
    critic.load_state_dict(torch.load('model/critic'))
else:

    initial(gen)
    initial(critic)

# Use Adam optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f'logs/real')
writer_fake = SummaryWriter(f'logs/fake')
writer_sample_digits = SummaryWriter(f'logs/sample_digits')

step = 0


gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(dataloader):
        # True images,noises and fake images  generated by noise
        real = real.to(device)

        labels = labels.to(device)

        # 加上一条
        BATCH_SIZE = real.shape[0]
        # Train critic
        # We need to train critic more so we need a for loop
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise, labels)

            # Warning:The batch_size of real and fake images are not equal
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)

            gp = gradient_penalty(critic, labels, real, fake, device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) \
                          + LAMBDA_GP * gp
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train the generator: -E[critic(gen_fake)]
        output = critic(fake, labels).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f'Epoch {epoch}|{NUM_EPOCHS} Batch {batch_idx}|{len(dataloader)}'
                f'Loss D {loss_critic:.4f},loss G {loss_gen:.4f}'
            )
            with torch.no_grad():
                fake = gen(noise, labels)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image('Real', img_grid_real, global_step=step)
                writer_real.add_image('Fake', img_grid_fake, global_step=step)

                # for cls in [0, 1, 2, 3, 4]:
                #     print(f'\r正在生成第{cls + 1}类', end='')
                #
                #     labels = torch.ones_like(labels).to(device)
                #     labels = labels * cls
                #     sample_digits = gen(noise, labels)
                #     img_sample_digits = \
                #         torchvision.utils.make_grid(sample_digits[:32], normalize=True)
                #     writer_sample_digits.add_image(f'Sample_digits {cls}', img_sample_digits, global_step=step)

                # 上面的等价写法
                fake_sampler=FakeSampler(gen,32)
                fake_sampler.gen_clses([0,1,2,3,4],writer_sample_digits,step)

                print()
            step += 1

torch.save(critic.state_dict(), 'model/critic')
torch.save(gen.state_dict(), 'model/gen')
