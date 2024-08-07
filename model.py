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


class ConditionalInstanceNorm2d(nn.Module):  # TODO train/test support
    # TODO this whole thing is probably very inefficient.
    def __init__(self, channels, c_dim, momentum=.1, epsilon=1e-8):
        super().__init__()
        self.momentum = momentum
        self.channels = channels
        self.epsilon = epsilon
        self.c_dim = c_dim
        self.register_buffer("running_mean", torch.zeros((2*c_dim, channels), requires_grad=False))
        self.register_buffer("running_std", torch.ones((2*c_dim, channels), requires_grad=False))
        
    def forward(self, im, c_trg, c_org):
        with torch.no_grad():
            std, mean = torch.std_mean(im, dim=(2,3))
            std += self.epsilon
            c_trg = c_trg.to(torch.bool)
            c_org = c_org.to(torch.bool)
            c_trg = torch.cat([c_trg, c_trg.logical_not()], dim=1)
            c_org = torch.cat([c_org, c_org.logical_not()], dim=1)
            for n in range(im.size(0)):
                for c in range(self.c_dim * 2):
                    if c_org[n, c]:
                        self.running_std[c] = self.running_std[c] * (1-self.momentum) + std[n] * self.momentum
                        self.running_mean[c] = self.running_mean[c] * (1-self.momentum) + mean[n] * self.momentum
        
            trg_std = torch.empty((im.size(0), self.channels), device=im.get_device(), requires_grad=False)
            trg_mean = torch.empty((im.size(0), self.channels), device=im.get_device(), requires_grad=False)
            
            for n in range(im.size(0)):
                trg_std[n] = self.running_std[c_trg[n]].mean(dim=0)
                trg_mean[n] = self.running_mean[c_trg[n]].mean(dim=0)
            
            
            def broadcast(x):
                return x.unsqueeze(2).unsqueeze(3).expand(-1, -1, im.size(2), im.size(3))
    
        return ((im-broadcast(mean)) / broadcast(std)) * broadcast(trg_std) + broadcast(trg_mean)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, poly_degree=3, poly_eps=.01):
        super(Generator, self).__init__()
        
        self.poly_degree = poly_degree
        self.poly_eps = poly_eps
        self.initial = nn.Sequential()
        self.initial.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.initial.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))        
        self.initial.append(nn.ReLU())

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            self.initial.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            self.initial.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            self.initial.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        self.initial.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, padding=1, bias=False))

        self.cond_norm = ConditionalInstanceNorm2d(curr_dim, c_dim)
        curr_dim += c_dim

        self.layers = nn.Sequential()
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

        self.layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=True))
        self.layers.append(nn.Tanh())

    def forward(self, im, c, c_org):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        
        x = self.initial(im)
        x = self.cond_norm(x, c, c_org)
        
        c_exp = c.view(c.size(0), c.size(1), 1, 1)
        c_exp = c_exp.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c_exp], dim=1)
        
        x = self.layers(x)
        return x


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