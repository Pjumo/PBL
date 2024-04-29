import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim

# Squueze & Excitation Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()

        # 입력의 평균 값을 1x1의 크기로 변환하는 것을 squeeze로 정의
        self.squeeze = nn.AdaptiveAvgPool2d((1,1)) 

        # 선형 변환 및 활성화 함수를 사용하여 입력된 값의 정보를 강조하는 것을 excitation으로 정의 
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels * r), # 입력 채널의 크기를 r배로 늘리는 선형 변환
            nn.SiLU(), # SiLU 활성화 함수를 적용
            nn.Linear(in_channels * r, in_channels), # 입력 채널의 크기로 줄이는 선형 변환 
            nn.Sigmoid() # Sigmoid 함수를 통해 함수를 적용하여 0과 1사이의 값으로 변환 
        )

    def forward(self, x):
        x = self.squeeze(x) # 입력을 squeeze하여 1x1로 변환
        x = x.view(x.size(0), -1) # 1x1의 입력을 벡터 형태로 변환
        x = self.excitation(x) # excitation 통과 
        x = x.view(x.size(0), x.size(1), 1, 1) # 강조 후 다시 1x1로 변환하여 저장
        return x

# EfficientNET에서 사용되는 MBConv정의 
class MBConv(nn.Module):
    expand = 6 # MBv6을 사용

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        
        # 첫 MBConv는 stochastic depth 사용하지 않음
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        # Residual Path 정의 
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * MBConv.expand, 1, stride=stride, padding=0, bias=False), # 1x1 컨볼루션 연산을 통해 입력 채널 수 확장
            nn.BatchNorm2d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3), 
            nn.SiLU(),
            nn.Conv2d(in_channels * MBConv.expand, in_channels * MBConv.expand, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2, bias=False, groups=in_channels*MBConv.expand),
            nn.BatchNorm2d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
            nn.SiLU()
        )

        self.se = SEBlock(in_channels * MBConv.expand, se_scale) # SE Block 적용

        self.project = nn.Sequential(
            nn.Conv2d(in_channels*MBConv.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )

        self.shortcut = (stride == 1) and (in_channels == out_channels)
    
    # 순전파
    def forward(self, x):
        # stochastic dropout 
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x= x_shortcut + x

        return x

# 첫 conv 정의
class MBvo(nn.Module):
    expand = 1 
   
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        
        # 첫 MBConv는 stochastic depth 사용하지 않음
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        # Residual Path 정의 
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels *MBvo.expand, 1, stride=stride, padding=0, bias=False), # 1x1 컨볼루션 연산을 통해 입력 채널 수 확장
            nn.BatchNorm2d(in_channels * MBvo.expand, momentum=0.99, eps=1e-3), 
            nn.SiLU(),
            nn.Conv2d(in_channels * MBvo.expand, in_channels * MBvo.expand, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2, bias=False, groups=in_channels*MBvo.expand),
            nn.BatchNorm2d(in_channels * MBvo.expand, momentum=0.99, eps=1e-3),
            nn.SiLU()
        )

        self.se = SEBlock(in_channels * MBvo.expand, se_scale) # SE Block 적용

        self.project = nn.Sequential(
            nn.Conv2d(in_channels*MBvo.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )

        self.shortcut = (stride == 1) and (in_channels == out_channels)
    
    # 순전파
    def forward(self, x):
        # stochastic dropout 
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x= x_shortcut + x

        return x


# EfficientNET 정의
class EfficientNet(nn.Module):
    def __init__(self, num_classes, width, depth, scale, dropout, se_scale, stochastic_depth=False, p=0.5):
        super().__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        kernel_size = [3, 3, 5, 3, 5, 5, 3]
        strides = [1, 2, 2, 2, 1, 2, 1]

        channels = [int(x*width) for x in channels]
        repeats = [int(x*depth) for x in repeats]

        # stochastic depth
        if stochastic_depth:
            self.p = p
            self.step = (1 - 0.5) / (sum(repeats) - 1)
        else:
            self.p = 1
            self.step = 0


        # efficient net
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, channels[0],3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0], momentum=0.99, eps=1e-3)
        )

        self.stage2 = self._make_Block(MBvo, repeats[0], channels[0], channels[1], kernel_size[0], strides[0], se_scale)

        self.stage3 = self._make_Block(MBConv, repeats[1], channels[1], channels[2], kernel_size[1], strides[1], se_scale)

        self.stage4 = self._make_Block(MBConv, repeats[2], channels[2], channels[3], kernel_size[2], strides[2], se_scale)

        self.stage5 = self._make_Block(MBConv, repeats[3], channels[3], channels[4], kernel_size[3], strides[3], se_scale)

        self.stage6 = self._make_Block(MBConv, repeats[4], channels[4], channels[5], kernel_size[4], strides[4], se_scale)

        self.stage7 = self._make_Block(MBConv, repeats[5], channels[5], channels[6], kernel_size[5], strides[5], se_scale)

        self.stage8 = self._make_Block(MBConv, repeats[6], channels[6], channels[7], kernel_size[6], strides[6], se_scale)

        self.stage9 = nn.Sequential(
            nn.Conv2d(channels[7], channels[8], 1, stride=1, bias=False),
            nn.BatchNorm2d(channels[8], momentum=0.99, eps=1e-3),
            nn.SiLU()
        ) 

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(channels[8], num_classes)

    def forward(self, x):
        x = self.upsample(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def _make_Block(self, block, repeats, in_channels, out_channels, kernel_size, stride, se_scale):
        strides = [stride] + [1] * (repeats - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, kernel_size, stride, se_scale, self.p))
            in_channels = out_channels
            self.p -= self.step

        return nn.Sequential(*layers)


def efficientnet_b0(num_classes):
    return EfficientNet(num_classes=num_classes, width=1.0, depth=1.0, scale=1.0,dropout=0.2, se_scale=4)

def efficientnet_b1(num_classes):
    return EfficientNet(num_classes=num_classes, width=1.0, depth=1.1, scale=240/224, dropout=0.2, se_scale=4)

def efficientnet_b2(num_classes):
    return EfficientNet(num_classes=num_classes, width=1.1, depth=1.2, scale=260/224., dropout=0.3, se_scale=4)

def efficientnet_b3(num_classes):
    return EfficientNet(num_classes=num_classes, width=1.2, depth=1.4, scale=300/224, dropout=0.3, se_scale=4)

def efficientnet_b4(num_classes):
    return EfficientNet(num_classes=num_classes, width=1.4, depth=1.8, scale=380/224, dropout=0.4, se_scale=4)

def efficientnet_b5(num_classes):
    return EfficientNet(num_classes=num_classes, width=1.6, depth=2.2, scale=456/224, dropout=0.4, se_scale=4)

def efficientnet_b6(num_classes):
    return EfficientNet(num_classes=num_classes, width=1.8, depth=2.6, scale=528/224, dropout=0.5, se_scale=4)

def efficientnet_b7(num_classes):
    return EfficientNet(num_classes=num_classes, width=2.0, depth=3.1, scale=600/224, dropout=0.5, se_scale=4)


