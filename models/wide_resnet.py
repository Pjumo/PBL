import torch
import torch.nn as nn
import numpy as np


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_channels = 16

        n = (depth-4)/6
        k = widen_factor



