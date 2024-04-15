import torch
import torch.nn as nn
import numpy as np


class WideBasic(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
        )


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_channels = 16

        n = (depth-4)/6
        k = widen_factor

        n_stages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, n_stages[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self.wide_layer(wide_basic)
