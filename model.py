import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm
import math
from torchvision.transforms import v2

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

class GenUpBlock(nn.Module):
    def __init__(self, channels, c_dim):
        super().__init__()
        self.im_to_channels = nn.Conv2d(3 + c_dim, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layers = nn.Sequential()
        self.layers.append(Block(2*channels, norm=True, updown="n"))
        self.layers.append(nn.Conv2d(2*channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.layers.append(nn.InstanceNorm2d(channels, affine=True))
        self.layers.append(Block(channels, norm=True, updown="u"))
        
    def forward(self, x, im, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, im.size(2), im.size(3))
        return self.layers(torch.cat([x, im, c], dim=1))

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        self.conv_dim = 128  # Just hacking it here.
        self.im_size = 128
        self.init_size = 8
        size = self.init_size
        
        self.up_blocks = nn.ModuleList()
        while size < self.im_size:
            self.up_blocks.append(GenUpBlock(self.conv_dim, c_dim))
            size *= 2

        self.out = nn.Sequential()
        self.out.append(nn.Conv2d(self.conv_dim, 3, kernel_size=7, stride=1, padding=3, bias=True))
        self.out.append(nn.Tanh())
        
    def forward(self, im, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        
        size = self.init_size
        x = torch.randn((im.size(0), self.conv_dim, self.init_size, self.init_size)).to(im.device())
        for b in self.up_blocks:
            im_resized = v2.Resize(im, size)
            x = b(x, im_resized)
            size *= 2
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

        self.conv1 = nn.Conv2d(conv_dim, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(conv_dim, c_dim, kernel_size=1, stride=1, padding=0, bias=False)
        spectral_norm(self.conv1)
        spectral_norm(self.conv2)
        
    def forward(self, x):
        h = self.layers(x) / self.num_residuals_factor
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))