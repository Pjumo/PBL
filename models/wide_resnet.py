import torch.nn as nn


class WideBasic(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.Dropout(dropout_rate),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
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
        out = x + out
        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_channels = 16

        n = (depth - 4) / 6
        k = widen_factor

        num_blocks = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, num_blocks[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self.wide_layer(WideBasic, num_blocks[1], n, dropout_rate, stride=1)
        self.layer2 = self.wide_layer(WideBasic, num_blocks[2], n, dropout_rate, stride=2)
        self.layer3 = self.wide_layer(WideBasic, num_blocks[3], n, dropout_rate, stride=2)
        self.avg_pool = nn.Sequential(
            nn.BatchNorm2d(num_blocks[3]),
            nn.ReLU(),
            nn.AvgPool2d(8)
        )
        self.linear = nn.Linear(num_blocks[3], num_classes)

    def wide_layer(self, block, channels, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, channels, dropout_rate, stride))
            self.in_channels = channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def WideResNet34(num_class):
    return WideResNet(34, 10, 0.3, num_class)
