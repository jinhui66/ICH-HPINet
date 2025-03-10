"""
modules.py - This file stores the rathering boring network blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from networks.STCN import mod_resnet
from networks.STCN import cbam


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, bn=False):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        # if bn:
        #     self.conv1 = nn.Sequential(
        #         nn.Conv2d(indim, outdim, kernel_size=3, padding=1), 
        #         nn.BatchNorm2d(outdim)
        #     )
        #     self.conv2 = nn.Sequential(
        #         nn.Conv2d(outdim, outdim, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(outdim)
        #     )
        # else:
        #     self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        #     self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = cbam.CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return x


class ValueEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=True, extra_chan=1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 64
        self.layer2 = resnet.layer2 # 1/8, 128
        self.layer3 = resnet.layer3 # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask):
        # key_f16 is the feature from the key encoder
        # image, mask: b*1*H*W

        f = torch.cat([image, mask], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 64
        x = self.layer2(x) # 1/8, 128
        x = self.layer3(x) # 1/16, 256

        x = self.fuser(x, key_f16)

        return x
 

class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # resnet = mod_resnet.resnet50(pretrained=True, extra_chan=0)
        # self.conv1 = resnet.conv1
        resnet = models.resnet50()
        self.conv1 = nn.Conv2d(1, resnet.conv1.out_channels, resnet.conv1.kernel_size, resnet.conv1.stride, resnet.conv1.padding, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        # self.skip_conv = nn.Sequential(
        #     nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(up_c)
        # )
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c, bn=True)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        # print("input_shape", skip_f.shape, up_f.shape)
        x = self.skip_conv(skip_f)
        # print("x_shape_3", x.shape)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        # print("x_shape_4", x.shape)
        x = self.out_conv(x)
        return x


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x):
        return self.key_proj(x)
