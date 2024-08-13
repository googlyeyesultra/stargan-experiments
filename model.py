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
        self.c_dim = c_dim

        self.downsample = nn.Sequential()
        self.downsample.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.downsample.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        self.downsample.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            self.downsample.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            self.downsample.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            self.downsample.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        self.class_nets = nn.ModuleList()
        for c in range(c_dim):
            class_net = nn.Sequential()
            self.class_nets.append(class_net)
            class_net.append(nn.Conv2d(curr_dim + 1, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
            class_net.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
            class_net.append(nn.ReLU(inplace=True))
            
            # Bottleneck layers.
            for i in range(repeat_num):
                class_net.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.upsample = nn.Sequential()
        # Up-sampling layers.
        for i in range(2):
            self.upsample.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            self.upsample.append(nn.Conv2d(curr_dim, curr_dim//2, kernel_size=5, padding=2))
            self.upsample.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            self.upsample.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.upsample.append(nn.Conv2d(curr_dim, 3 * (poly_degree+1), kernel_size=7, stride=1, padding=3, bias=True))

    def forward(self, im, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        
        downed = self.downsample(im)
        result_sum = torch.zeros_like(downed)
        for i in range(self.c_dim):
            c_selected = c[:,i]
            c_selected = c_selected.view(c_selected.size(0), 1, 1, 1)
            c_selected = c_selected.expand(c_selected.size(0), 1, downed.size(2), downed.size(3))
            stacked = torch.cat([downed, c_selected], dim=1)
            result = self.class_nets[i](stacked)
            result_sum += result
        
        x = self.upsample(result_sum)

        num = x.unflatten(dim=1, sizes=(self.poly_degree+1, 3))
        denom = num.abs().sum(dim=1, keepdim=True) + self.poly_eps
        coeffs = num / denom

        pows = torch.stack([im.pow(i) for i in range(self.poly_degree+1)], dim=1)
        return (pows * coeffs).sum(1)


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
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        spectral_norm(self.conv1)
        spectral_norm(self.conv2)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))