import torch
import torch.nn as nn
import numpy as np

from . import up_or_down_sampling
from layers import dense, conv2d

class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        downsample=False,
        act = nn.LeakyReLU(0.2),
        fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()
     
        self.fir_kernel = fir_kernel
        self.downsample = downsample
        
        self.conv1 = nn.Sequential(conv2d(in_channel, out_channel, kernel_size, padding=padding),)
        self.conv2 = nn.Sequential(conv2d(out_channel, out_channel, kernel_size, padding=padding,init_scale=0.))
        self.act = act
        self.skip = nn.Sequential(conv2d(in_channel, out_channel, 1, padding=0, bias=False),)
        
    def forward(self, input):
        out = self.act(input)
        out = self.conv1(out)
        out = self.act(out)
        
        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)
        return out

class Discriminator_small(nn.Module):
  """A discriminator for small images (CIFAR10, StackMNIST)."""

  def __init__(self, nc = 3, ngf = 64, act=nn.LeakyReLU(0.2)):
    super().__init__()
    
    self.act = act
    
    # Encoding layers where the resolution decreases
    self.start_conv = conv2d(nc,ngf*2,1, padding=0)
    self.conv1 = DownConvBlock(ngf*2, ngf*2, act=act)
    self.conv2 = DownConvBlock(ngf*2, ngf*4, downsample=True, act=act)
    self.conv3 = DownConvBlock(ngf*4, ngf*8, downsample=True, act=act)
    self.conv4 = DownConvBlock(ngf*8, ngf*8, downsample=True, act=act)
    self.final_conv = conv2d(ngf*8, ngf*8, 3, padding=1, init_scale=0.)
    self.end_linear = dense(ngf*8, 1)
    self.stddev_group = 4
    self.stddev_feat = 1
    
        
  def forward(self, input_x):
    h = self.start_conv(input_x)
    h = self.conv1(h)
    h = self.conv2(h)
    h = self.conv3(h)
    out = self.conv4(h)
    
    out = self.final_conv(out)
    out = self.act(out)
    out = out.view(out.shape[0], out.shape[1], -1).sum(2)
    out = self.end_linear(out)
    return out

class TransformNet(nn.Module):
    def __init__(self, size):
        super(TransformNet, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size, self.size))

    def forward(self, input):
        out = self.net(input)
        return out / torch.sqrt(torch.sum(out ** 2, dim=1, keepdim=True))