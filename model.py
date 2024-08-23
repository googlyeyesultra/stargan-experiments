import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm
import math

class Block(nn.Module):
    def __init__(self, channels, norm=False, sn=False, updown="n"):
        super().__init__()
        
        self.layers = nn.Sequential()
        conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=not norm)
        if sn:
            spectral_norm(conv1)
        self.layers.append(conv1)
        if norm:
            self.layers.append(nn.InstanceNorm2d(channels, affine=True))
        self.layers.append(nn.LeakyReLU(.3))
        
        if updown == "d":
            conv2 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=not norm)
            self.skip = nn.Conv2d(channels, channels, kernel_size=2, stride=2, padding=0, bias=not norm)
            if sn:
                spectral_norm(self.skip)
        elif updown == "u":
            self.layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=not norm)
            self.skip = nn.Upsample(scale_factor=2, mode="bilinear")
        else:
            conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=not norm)
            self.skip = nn.Identity()
        
        if sn:
            spectral_norm(conv2)
        self.layers.append(conv2)
        if norm:
            self.layers.append(nn.InstanceNorm2d(channels, affine=True))

    def forward(self, x):
        return self.skip(x) + self.layers(x)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        self.layers = nn.Sequential()
        conv_dim = 128  # Just hacking it here.
        
        self.layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))

        # Down-sampling layers.
        self.layers.append(Block(conv_dim, norm=True, updown="d"))
        self.layers.append(Block(conv_dim, norm=True, updown="n"))
        self.layers.append(Block(conv_dim, norm=True, updown="d"))

        # Bottleneck layers.
        for i in range(repeat_num):
            self.layers.append(Block(conv_dim, norm=True, updown="n"))

        # Up-sampling layers.
        self.layers.append(Block(conv_dim, norm=True, updown="u"))
        self.layers.append(Block(conv_dim, norm=True, updown="n"))
        self.layers.append(Block(conv_dim, norm=True, updown="u"))
        
        self.layers.append(Block(conv_dim, norm=True, updown="n"))
        self.layers.append(Block(conv_dim, norm=True, updown="n"))
        self.layers.append(nn.Conv2d(conv_dim, 3, kernel_size=7, stride=1, padding=3, bias=True))
        self.layers.append(nn.Tanh())
        
    def forward(self, im, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, im.size(2), im.size(3))
        x = torch.cat([im, c], dim=1)
        x = self.layers(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super().__init__()
        
        conv_dim = 128  # Just hacking it here.
        self.layers = nn.Sequential()
        conv = nn.Conv2d(3, conv_dim, kernel_size=3, stride=1, padding=1)
        spectral_norm(conv)
        self.layers.append(conv)

        down_layers = int(math.log2(image_size))

        for i in range(down_layers):
            self.layers.append(Block(conv_dim, sn=True, updown="n"))
            self.layers.append(Block(conv_dim, sn=True, updown="d"))

        for i in range(3):
            self.layers.append(Block(conv_dim, sn=True, updown="n"))     

        self.num_residuals_factor = 2 ** (down_layers + 3)

        self.conv1 = nn.Conv2d(conv_dim, conv_dim, kernel_size=1, stride=1, padding=0, bias=True)
        spectral_norm(self.conv1)
        
        self.labels_ff = nn.Sequential()
        lin = nn.Linear(conv_dim + c_dim, conv_dim)
        spectral_norm(lin)
        self.labels_ff.append(lin)
        self.labels_ff.append(nn.LeakyReLU(.2))
        for i in range(10):
            lin = nn.Linear(conv_dim, conv_dim, bias=False)
            spectral_norm(lin)
            self.labels_ff.append(lin)
            norm = nn.BatchNorm1d(conv_dim, affine=True)
            spectral_norm(norm)
            self.labels_ff.append(norm)
            self.labels_ff.append(nn.LeakyReLU(.2))
            
        self.labels_ff.append(nn.Linear(conv_dim, 1))
        
    def forward(self, x, c):
        x = self.layers(x).squeeze(dim=(2, 3)) / self.num_residuals_factor
        x = torch.cat([x, c], dim=1)
        return self.labels_ff(x)