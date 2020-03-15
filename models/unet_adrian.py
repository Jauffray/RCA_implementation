# Inspired on https://github.com/jvanvugt/pytorch-unet
# Improvements added by A. Galdran (Dec. 2019)

import torch
from torch import nn


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool='max_pool'):
        '''
        pool can be False (no pooling), 'maxpool', 'strided_conv', or 'strided_conv_replace'
        '''
        super(ConvBlock, self).__init__()
        stride = 1
        pad = (kernel_size - 1) // 2
        block = []

        if pool == 'max_pool':
            block.append(nn.MaxPool2d(kernel_size=2))
        elif pool == 'strided_conv':
            block.append(nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2))
        elif pool == 'strided_conv_replace':
            stride = 2
        elif pool == False:
            pass
        else:
            raise Exception('Unsupported pool type')

        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_channels))

        block.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, up_mode='transp_conv'):
        super(UpsampleBlock, self).__init__()
        block = []
        if up_mode == 'transp_conv':
            block.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
        elif up_mode=='up_conv':
            block.append(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False))
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            raise Exception('Upsampling mode not supported')
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UpConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, up_mode='transp_conv', kernel_size=3):
        super(UpConvBlock, self).__init__()

        self.up_layer = UpsampleBlock(in_channels, out_channels, up_mode=up_mode)
        self.conv_layer = ConvBlock(in_channels, out_channels, pool=False)  # SEE HERE

    def forward(self, x, skip):
        up = self.up_layer(x)
        out = torch.cat([up, skip], dim=1)
        out = self.conv_layer(out)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, layers, kernel_size=3, up_mode='transp_conv', pool_mode='max_pool'):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.down_path = nn.ModuleList()
        layers.insert(0, in_channels)
        for i in range(len(layers) - 1):
            #         If we are using the ConvBlock handling pooling
            block = ConvBlock(in_channels=layers[i], out_channels=layers[i + 1],
                              kernel_size=kernel_size,
                              pool=False if i == 0 else pool_mode)  # no pool in first layer
            self.down_path.append(block)

        self.up_path = nn.ModuleList()
        layers = layers[-1: 0: -1]  # remove first and reverse

        for i in range(len(layers) - 1):
            block = UpConvBlock(in_channels=layers[i], out_channels=layers[i + 1], up_mode=up_mode)
            self.up_path.append(block)

        self.final = nn.Conv2d(layers[-1], n_classes, kernel_size=1)

    def forward(self, x):
        down_activations = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                # we do not need the last activation in the bottom
                down_activations.append(x)

        down_activations.reverse()
        for i, up in enumerate(self.up_path):
            x = up(x, down_activations[i])
        x = self.final(x)
        return x