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


class SEBlock(nn.Module):
    def __init__(self, size, channels, downsample=False):
        super().__init__()
        self.size = size
        self.ff = nn.Sequential()
        self.ff.append(nn.Linear(channels, channels))
        self.ff.append(nn.LeakyReLU(.1))
        self.ff.append(nn.Linear(channels, channels))
        self.ff.append(nn.Sigmoid())
        
        self.convnet = nn.Sequential()
        conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        spectral_norm(conv1)
        self.convnet.append(conv1)
        self.convnet.append(nn.LeakyReLU(.1))
        if downsample:
            self.trg_size = size // 2
            conv2 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
            self.ds_skip = nn.Conv2d(channels, channels, kernel_size=2, stride=2, padding=0)
            spectral_norm(self.ds_skip)
        else:
            self.trg_size = size
            conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
            self.ds_skip = nn.Identity()
        
        spectral_norm(conv2)
        self.convnet.append(conv2)

    def forward(self, x):
        squeezed = F.adaptive_avg_pool2d(x, (1, 1))
        squeezed = self.ff(squeezed)
        return self.ds_skip(x) + self.convnet(x) * squeezed.expand(-1, -1, self.trg_size, self.trg_size)

class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super().__init__()
        conv_dim = 128  # Just hacking this in since we're using conv_dim differently.
        
        self.layers = nn.Sequential()
        initial = nn.Conv2d(3, conv_dim, kernel_size=3, stride=1, padding=1)
        spectral_norm(initial)
        self.layers.append(initial)
        size = image_size
        for i in range(repeat_num):
            self.layers.append(SEBlock(size, conv_dim))
            self.layers.append(SEBlock(size, conv_dim, True))
            size //=2
            
        kernel_size = int(image_size / np.power(2, repeat_num))
        self.conv1 = nn.Conv2d(conv_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(conv_dim, c_dim, kernel_size=kernel_size, bias=False)
        spectral_norm(self.conv1)
        spectral_norm(self.conv2)
        
    def forward(self, x):
        h = self.layers(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))