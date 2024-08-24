import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm
import math

class Block(nn.Module):
    def __init__(self, channels, norm=False, sn=False, leaky=True, updown="n"):
        super().__init__()
        
        activ = nn.LeakyReLU(.3) if leaky else nn.ReLU(inplace=True)
        
        self.layers = nn.Sequential()
        conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=not norm)
        if sn:
            spectral_norm(conv1)
        self.layers.append(conv1)
        if norm:
            self.layers.append(nn.InstanceNorm2d(channels, affine=True))
        self.layers.append(activ)
        
        if updown == "d":
            conv2 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=not norm)
        elif updown == "u":
            self.layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=not norm)
        else:
            conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=not norm)
        
        if sn:
            spectral_norm(conv2)
        self.layers.append(conv2)
        if norm:
            self.layers.append(nn.InstanceNorm2d(channels, affine=True))
            
        self.layers.append(activ)
        self.residual = updown == "n"

    def forward(self, x):
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        self.layers = nn.Sequential()
        conv_dim = 128  # Just hacking it here.
        
        self.layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))

        # Down-sampling layers.
        self.layers.append(Block(conv_dim, norm=True, leaky=False, updown="d"))
        self.layers.append(Block(conv_dim, norm=True, leaky=False, updown="n"))
        self.layers.append(Block(conv_dim, norm=True, leaky=False, updown="d"))

        # Bottleneck layers.
        for i in range(repeat_num):
            self.layers.append(Block(conv_dim, norm=True, leaky=False, updown="n"))

        # Up-sampling layers.
        self.layers.append(Block(conv_dim, norm=True, leaky=False, updown="u"))
        self.layers.append(Block(conv_dim, norm=True, leaky=False, updown="n"))
        self.layers.append(Block(conv_dim, norm=True, leaky=False, updown="u"))
        
        self.final = nn.Sequential()
        self.final.append(nn.Conv2d(3+conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))
        self.final.append(nn.InstanceNorm2d(conv_dim, affine=True))
        self.final.append(nn.ReLU(inplace=True))
        self.final.append(Block(conv_dim, norm=True, leaky=False, updown="n"))
        self.final.append(Block(conv_dim, norm=True, leaky=False, updown="n"))
        self.final.append(Block(conv_dim, norm=True, leaky=False, updown="n"))
        self.final.append(Block(conv_dim, norm=True, leaky=False, updown="n"))
        self.final.append(nn.Conv2d(conv_dim, 3, kernel_size=7, stride=1, padding=3, bias=True))
        self.final.append(nn.Tanh())
        
    def forward(self, im, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, im.size(2), im.size(3))
        x = torch.cat([im, c], dim=1)
        x = self.layers(x)
        x = torch.cat([x, im], dim=1)
        return self.final(x)

class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super().__init__()
        
        conv_dim = 128  # Just hacking it here.
        self.layers = nn.Sequential()
        conv = nn.Conv2d(3 + c_dim, conv_dim, kernel_size=3, stride=1, padding=1)
        spectral_norm(conv)
        self.layers.append(conv)

        down_layers = int(math.log2(image_size))

        for i in range(down_layers):
            self.layers.append(Block(conv_dim, sn=True, updown="n"))
            self.layers.append(Block(conv_dim, sn=True, updown="d"))

        for i in range(5):
            self.layers.append(Block(conv_dim, sn=True, updown="n"))     

        self.num_residuals_factor = 2 ** (down_layers + 5)

        self.conv1 = nn.Conv2d(conv_dim, 1, kernel_size=1, stride=1, padding=0, bias=True)
        spectral_norm(self.conv1)
        
        self.labels_ff = nn.Sequential()
        self.labels_ff.append(nn.Linear(c_dim, 64))
        self.labels_ff.append(nn.ReLU(inplace=True))
        self.labels_ff.append(nn.Linear(64, c_dim))
        
    def forward(self, x, c):
        c = self.labels_ff(c)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.layers(x).squeeze(dim=(1,2,3)) / self.num_residuals_factor