#!/usr/bin/python
# author eson


import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super().__init__()
        self.net = nn.Sequential(

            self._block(z_dim, features_g * 16, 4, 1, 0),  # z_dim x 1 x 1-> 1024 x 4 x 4 Maybe stride=2 also work?
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 512 x 8 x 8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 256 x 16 x 16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 128 x 32 x 32
            nn.ConvTranspose2d(features_g * 2, channels_img, 4, 2, 1),  # 64 x 64 x 64
            nn.Tanh()  # [-1,1]

        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, channels_imgs, features_d):
        super().__init__()
        self.disc = nn.Sequential(
            # input 3 x 64 x 64
            nn.Conv2d(channels_imgs, features_d, kernel_size=4, stride=2, padding=1),
            # 64 x 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            # 128 x 16 x 16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            # 256 x 8 x 8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # 512 x 4 x 4
            nn.Conv2d(features_d * 8, 1, 4, 2, 0),
            # 1 x 1 x 1
            #

        )

    def _block(self, in_channels, out_channels, kernal_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels,affine=True),  # LayerNorm-->InstanceNorm
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


def initial(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0, 0.02)


if __name__ == '__main__':
    # g=Generator(100,3,64)
    # print([m for m in g.modules()])
    # print(len([m for m in g.modules()]))
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, features_d=8)
    initial(disc)
    assert disc(x).shape == (N, 1, 1, 1)

    gen = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print('Success.')
