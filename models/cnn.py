import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):  # ResNet을 구성하는 Residual Block 구조 구현
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = F.relu(out)
        return out


class CNN(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(CNN, self).__init__()
        self.in_channels = 64
        self.base = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

        self.layer1 = self.make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []

        for stride in strides:
            block = ResidualBlock(self.in_channels, out_channels, stride)
            layers.append(block)
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)

        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out


def CNN18(num_classes):
    return CNN([2, 2, 2, 2], num_classes)


def CNN34(num_classes):
    return CNN([3, 4, 6, 3], num_classes)
