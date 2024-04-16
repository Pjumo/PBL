import torch.nn as nn


class ResNextBottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        groups = 32

        num_depth = int(self.expansion * out_channels / 64)  # 그룹당 채널 수(depth per group)
        self.split_block = nn.Sequential(
            nn.Conv2d(in_channels, groups * num_depth, kernel_size=1, stride=1, groups=groups, bias=False),
            nn.BatchNorm2d(groups * num_depth),
            nn.ReLU(),
            nn.Conv2d(groups * num_depth, groups * num_depth, kernel_size=3, stride=stride, groups=groups, padding=1,
                      bias=False),
            nn.BatchNorm2d(groups * num_depth),
            nn.ReLU(),
            nn.Conv2d(groups * num_depth, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(),
        )

        if self.stride != 1 or self.in_channels != self.out_channels * self.expansion:
            self.down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.out_channels * self.expansion),
                nn.ReLU()
            )

    def forward(self, x):
        out = self.split_block(x)
        if self.stride != 1 or self.in_channels != self.out_channels * self.expansion:
            x = self.down_sample(x)
        out = x + out
        return out


class ResNext(nn.Module):
    def __init__(self, block_, num_blocks, num_classes=10):
        super(ResNext, self).__init__()
        self.block_ = block_
        self.in_channels = 64

        self.base = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(512, num_blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
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
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)

        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out


def resnext50(num_classes):
    return ResNext(ResNextBottleNeck, [3, 4, 6, 3], num_classes)


def resnext101(num_classes):
    return ResNext(ResNextBottleNeck, [3, 4, 23, 3], num_classes)


def resnext152(num_classes):
    return ResNext(ResNextBottleNeck, [3, 4, 36, 3], num_classes)
