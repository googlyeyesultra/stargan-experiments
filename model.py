import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, poly_degree=3, poly_eps=.01):
        super(Generator, self).__init__()
        
        self.poly_degree = poly_degree
        self.poly_eps = poly_eps

        self.layers = nn.Sequential()
        self.layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        self.layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            self.layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            self.layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            self.layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            self.layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            self.layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            self.layers.append(nn.Conv2d(curr_dim, curr_dim//2, kernel_size=5, padding=2))
            self.layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            self.layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.layers.append(nn.Conv2d(curr_dim, 3 * (poly_degree+1), kernel_size=7, stride=1, padding=3, bias=True))

    def forward(self, im, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, im.size(2), im.size(3))
        x = torch.cat([im, c], dim=1)
        x = self.layers(x)

        num = x.unflatten(dim=1, sizes=(self.poly_degree+1, 3))
        denom = num.abs().sum(dim=1, keepdim=True) + self.poly_eps
        coeffs = num / denom

        pows = torch.stack([im.pow(i) for i in range(self.poly_degree+1)], dim=1)
        return (pows * coeffs).sum(1)


class DiscrimBlock(nn.Module):
    def __init__(self, conv_dim, num_convs=3):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(num_convs):
            conv = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
            spectral_norm(conv)
            self.layers.append(conv)
            if i != num_convs - 1:
                self.layers.append(nn.LeakyReLU(.01, inplace=True))
        
        self.final_activ = nn.LeakyReLU(.01)
        
    def forward(self, x):
        return self.final_activ(self.layers(x) + x)


class DownsampleBlock(nn.Module):
    def __init__(self, conv_dim, num_convs=3):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(num_convs):
            conv = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
            spectral_norm(conv)
            self.layers.append(conv)
            if i != num_convs - 1:
                self.layers.append(nn.LeakyReLU(.01, inplace=True))
        
        self.final = nn.Sequential()
        conv = nn.Conv2d(conv_dim, conv_dim, kernel_size=4, stride=2, padding=1, bias=True)
        spectral_norm(conv)
        self.final.append(conv)
        self.final.append(nn.LeakyReLU(.01))
        
    def forward(self, x):
        return self.final(self.layers(x) + x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    # Leaving repeat num in for now so I don't need to change solver/main.
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super().__init__()
        self.layers = nn.Sequential()
        
        initial_conv = nn.Conv2d(3, conv_dim, kernel_size=3, stride=1, padding=1)
        spectral_norm(initial_conv)
        self.layers.append(initial_conv)
        self.layers.append(nn.LeakyReLU(.01))
        
        size = image_size
        while size > 1:
            self.layers.append(DiscrimBlock(conv_dim))
            self.layers.append(DiscrimBlock(conv_dim))
            self.layers.append(DownsampleBlock(conv_dim))
            size //= 2
            
        self.conv1 = nn.Conv2d(conv_dim, 1, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(conv_dim, c_dim, kernel_size=1, bias=True)
        spectral_norm(self.conv1)
        spectral_norm(self.conv2)
        
    def forward(self, x):
        x = self.layers(x)
        return self.conv1(x).squeeze(1, 2, 3), self.conv2(x).squeeze(2, 3)