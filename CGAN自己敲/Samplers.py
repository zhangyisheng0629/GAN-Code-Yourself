#!/usr/bin/python
# author eson
import numpy as np
import torch

import torchvision


device='cuda'

class FakeSampler:
    # 初始化：取样的模型，每次取多少个样本
    def __init__(self, model, num_gen):
        self.model = model
        self.num_gen = num_gen
        pass

    def gen_cls(self, target,Z_DIM=100):
        noise = torch.rand((self.num_gen, Z_DIM, 1, 1)).to(device)
        labels = target * torch.ones(self.num_gen).int().to(device)
        sample_images = self.model(noise, labels)
        img_grid = torchvision.utils.make_grid(sample_images, normalize=True)
        return img_grid

    # Given a class index list then gen images of these classes
    def gen_clses(self, gen_list, writer, step):
        for cls in gen_list:
            print(f'\r正在生成类别索引为 {cls} 的图片', end='')
            img_grid = self.gen_cls(cls)
            writer.add_image(f'Fake image {cls} ', img_grid, global_step=step)


class TrueSampler:
    def __init__(self, dataset, num, transforms):
        self.num = num
        self.dataset = dataset
        self.transforms = transforms
        pass

    def gen_cls(self, cls):
        imgs = np.array([sample.transpose(2, 0, 1)
                         for sample, target in zip(self.dataset.data, self.dataset.targets) if target == cls][:32])
        imgs = imgs.astype('float32')
        imgs = torch.tensor(imgs, )
        imgs = self.transforms(imgs)
        imgs_grid = torchvision.utils.make_grid(imgs, normalize=True)
        return imgs_grid

    def gen_clses(self, gen_list, writer):
        for cls in gen_list:
            imgs_grid = self.gen_cls(cls)
            writer.add_image(f'Real_per_cls {cls}', imgs_grid)

    # for data in dataset:
    #
    #     sample_images=
    #     img_grid=torchvision.utils.make_grid(sample_images,normalize=True)


if __name__ == '__main__':
    pass
