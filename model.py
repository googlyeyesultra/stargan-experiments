import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm, weight_norm


# Simplified adaptation of modulated convolution in https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
class ModConv(nn.Module):  # Modulated convolution like StyleGAN 2.
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, padding, stride=1):
        super().__init__()
        self.eps = 1e-8
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        
        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / fan_in ** .5
        self.padding = padding
        self.stride = stride
        
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.modulation = nn.Linear(style_dim, in_channel)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        
        # weight_norm(self.weight)  # Can't weight norm this
        # weight_norm(self.modulation)
        
    def forward(self, x, style):
        batch, in_channel, height, width = x.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
        weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        
        x = x.view(1, batch * in_channel, height, width)
        #padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="reflect")
        out = F.conv2d(x, weight, groups=batch, padding=self.padding, stride=self.stride)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)
        return out + self.bias


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim):
        super(ResidualBlock, self).__init__()
        self.c1 = ModConv(dim_in, dim_out, kernel_size=3, style_dim=style_dim, padding=1)
        self.activ = nn.ReLU(inplace=True)
        self.c2 = ModConv(dim_out, dim_out, kernel_size=3, style_dim=style_dim, padding=1)
        
    def forward(self, x, style):
        f = self.c1(x, style)
        f = self.activ(f)
        f = self.c2(f, style)
        return x + f

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()
        
        style_dim = 64

        self.down = nn.ModuleList()
        self.down.append(ModConv(3, conv_dim, kernel_size=7, padding=3, style_dim=style_dim))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            self.down.append(ModConv(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, style_dim=style_dim))
            curr_dim = curr_dim * 2

        self.modlayers = nn.ModuleList()
        # Bottleneck layers.
        for i in range(repeat_num):
            self.modlayers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, style_dim=style_dim))

        self.up = nn.Sequential()
        # Up-sampling layers.
        for i in range(2):
            self.up.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            c = nn.Conv2d(curr_dim, curr_dim//2, kernel_size=5, padding=2, padding_mode="reflect")
            weight_norm(c)
            self.up.append(c)
            self.up.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.final_res = nn.ModuleList()
        for i in range(3):
            self.final_res.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, style_dim=style_dim))

        self.final = nn.Sequential()
        c = nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
        weight_norm(c)
        self.final.append(c)
        self.final.append(nn.Tanh())
        
        
        self.style_net = nn.Sequential()
        l = nn.Linear(c_dim, style_dim)
        weight_norm(l)
        self.style_net.append(l)
        self.style_net.append(nn.ReLU(inplace=True))
        for i in range(5):
            l = nn.Linear(style_dim, style_dim)
            weight_norm(l)
            self.style_net.append(l)
            self.style_net.append(nn.ReLU(inplace=True))

    def forward(self, x, c, orig_labels):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        
        c = c - orig_labels
        style = self.style_net(c)
        
        for l in self.down:
            x = l(x, style)
            x = F.relu(x, inplace=True)
        
        for l in self.modlayers:
            x = l(x, style)
            
        x = self.up(x)
        
        for l in self.final_res:
            x = l(x, style)
            
        return self.final(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        
        self.style_net = nn.Sequential()
        style_dim = 64
        l = nn.Linear(c_dim, style_dim)
        spectral_norm(l)
        self.style_net.append(l)
        self.style_net.append(nn.ReLU(inplace=True))
        for i in range(5):
            l = nn.Linear(style_dim, style_dim)
            spectral_norm(l)
            self.style_net.append(l)
            self.style_net.append(nn.ReLU(inplace=True))
        
        self.layers = nn.ModuleList()
        conv = ModConv(3, conv_dim, kernel_size=4, style_dim=style_dim, stride=2, padding=1)
        spectral_norm(conv)
        self.layers.append(conv)

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            conv = ModConv(curr_dim, curr_dim*2, kernel_size=4, style_dim=style_dim, stride=2, padding=1)
            spectral_norm(conv)
            self.layers.append(conv)
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        conv = ModConv(curr_dim, 1, kernel_size=kernel_size, style_dim=style_dim, stride=1, padding=0)
        spectral_norm(conv)
        self.layers.append(conv)

        
    def forward(self, x, labels):
        style = self.style_net(labels)
        
        for l in self.layers:
            x = l(x, style)
            x = F.leaky_relu_(x, .01)
        
        return x.squeeze(dim=(2, 3))