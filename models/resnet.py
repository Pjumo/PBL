import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):  # ResNet을 구성하는 Residual Basic Block 구조 구현
    expansion = 1

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

        if self.stride != 1 or self.in_channels != self.out_channels:
            self.down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )

    def forward(self, x):
        out = self.conv_block(x)
        if self.stride != 1 or self.in_channels != self.out_channels:
            x = self.down_sample(x)
        out = F.relu(x + out)
        return out


class PreActBlock(nn.Module):  # Residual Basic Block Pre-Act 방식 구현
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

        if self.stride != 1 or self.in_channels != self.out_channels:
            self.down_sample = nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.conv_block(x)
        if self.stride != 1 or self.in_channels != self.out_channels:
            x = self.down_sample(x)
        out = x + out
        return out


class Bottleneck(nn.Module):  # ResNet을 구성하는 Residual Bottleneck Block 구조 구현
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

        if self.stride != 1 or self.in_channels != self.out_channels * 4:
            self.down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.out_channels * 4)
            )

    def forward(self, x):
        out = self.conv_block(x)
        if self.stride != 1 or self.in_channels != self.out_channels * 4:
            x = self.down_sample(x)
        out = F.relu(x + out)
        return out


class PreActBottleneck(nn.Module):  # Residual Bottleneck Block Pre-Act 방식 구현
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBottleneck, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False),
        )

        if self.stride != 1 or self.in_channels != self.out_channels * 4:
            self.down_sample = nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, self.out_channels * 4, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.conv_block(x)
        if self.stride != 1 or self.in_channels != self.out_channels * 4:
            x = self.down_sample(x)
        out = x + out
        return out


class ResNet(nn.Module):
    def __init__(self, block_, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.block_ = block_
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
        self.fc = nn.Linear(512 * block_.expansion, num_classes)

    def make_layer(self, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []

        for stride in strides:
            block = self.block_(self.in_channels, out_channels, stride)
            layers.append(block)
            self.in_channels = out_channels * block.expansion
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


def ResNet18(num_classes):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes):
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def PreActResNet18(num_classes):
    return ResNet(PreActBlock, [2, 2, 2, 2], num_classes)


def PreActResNet34(num_classes):
    return ResNet(PreActBlock, [3, 4, 6, 3], num_classes)


def PreActResNet50(num_classes):
    return ResNet(PreActBottleneck, [3, 4, 6, 3], num_classes)


def PreActResNet101(num_classes):
    return ResNet(PreActBottleneck, [3, 4, 23, 3], num_classes)


def PreActResNet152(num_classes):
    return ResNet(PreActBottleneck, [3, 8, 36, 3], num_classes)
