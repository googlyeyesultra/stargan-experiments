import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm, weight_norm


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential()
        c = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
        weight_norm(c)
        self.main.append(c)
        self.main.append(nn.ReLU(inplace=True))
        c2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1)
        weight_norm(c2)
        self.main.append(c2)
        
    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        self.layers = nn.Sequential()
        c = nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3)
        weight_norm(c)
        self.layers.append(c)
        self.layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            c = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
            weight_norm(c)
            self.layers.append(c)
            self.layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            self.layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            self.layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            c = nn.Conv2d(curr_dim, curr_dim//2, kernel_size=5, padding=2)
            weight_norm(c)
            self.layers.append(c)
            self.layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        c = nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3)
        weight_norm(c)
        self.layers.append(c)
        self.layers.append(nn.Tanh())

    def forward(self, im, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, im.size(2), im.size(3))
        x = torch.cat([im, c], dim=1)
        return self.layers(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        conv = nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)
        spectral_norm(conv)
        layers.append(conv)
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            conv = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
            spectral_norm(conv)
            layers.append(conv)
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        conv = nn.Conv2d(curr_dim, c_dim*2, kernel_size=kernel_size, stride=1, padding=1, bias=True)
        spectral_norm(conv)
        layers.append(conv)
        self.main = nn.Sequential(*layers)

        
    def forward(self, x, labels):
        h = self.main(x)
        labels = torch.cat([labels, 1-labels], dim=1).to(torch.bool)
        return h[labels].mean(dim=1)