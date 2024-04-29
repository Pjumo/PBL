# 출처: https://github.com/hankyul2/EfficientNetV2-pytorch

import copy
from functools import partial
from collections import OrderedDict

import torch
from torch import nn

# from .pretrained_weight_loader import load_from_zoo

# Convolution-Normalization-Activation Module
class ConvBNAct(nn.Sequential):

    def __init__(self, in_channel, out_channel, kernel_size, stride, groups, norm_layer, act, conv_layer=nn.Conv2d):
        super(ConvBNAct, self).__init__(
            conv_layer(in_channel, out_channel, kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=groups, bias=False),
            norm_layer(out_channel),
            act()
        )

# Squeeze-Excitation Unit
class SEUnit(nn.Module):
  
    def __init__(self, in_channel, reduction_ratio=4, act1=partial(nn.SiLU, inplace=True), act2=nn.Sigmoid):
        super(SEUnit, self).__init__()
        hidden_dim = in_channel // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channel, hidden_dim, (1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, in_channel, (1, 1), bias=True)
        self.act1 = act1()
        self.act2 = act2()

    def forward(self, x):
        return x * self.act2(self.fc2(self.act1(self.fc1(self.avg_pool(x)))))

class StochasticDepth(nn.Module):
    
    def __init__(self, prob, mode):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        if self.prob == 0.0 or not self.training:
            return x
        else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == 'row' else [1]
            return x * torch.empty(shape).bernoulli_(self.survival).div_(self.survival).to(x.device)

# EfficientNet Building block configuration
class MBConvConfig:
    
    def __init__(self, expand_ratio: float, kernel: int, stride: int, in_ch: int, out_ch: int, layers: int,
                 use_se: bool, fused: bool, act=nn.SiLU, norm_layer=nn.BatchNorm2d):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_layers = layers
        self.act = act
        self.norm_layer = norm_layer
        self.use_se = use_se
        self.fused = fused

    @staticmethod
    def adjust_channels(channel, factor, divisible=8):
        new_channel = channel * factor
        divisible_channel = max(divisible, (int(new_channel + divisible / 2) // divisible) * divisible)
        divisible_channel += divisible if divisible_channel < 0.9 * new_channel else 0
        return divisible_channel
    
# EfficientNet main building blocks
class MBConv(nn.Module):
    
    def __init__(self, c, sd_prob=0.0):
        super(MBConv, self).__init__()
        inter_channel = c.adjust_channels(c.in_ch, c.expand_ratio)
        block = []

        if c.expand_ratio == 1:
            block.append(('fused', ConvBNAct(c.in_ch, inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)))
        elif c.fused:
            block.append(('fused', ConvBNAct(c.in_ch, inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)))
            block.append(('fused_point_wise', ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))
        else:
            block.append(('linear_bottleneck', ConvBNAct(c.in_ch, inter_channel, 1, 1, 1, c.norm_layer, c.act)))
            block.append(('depth_wise', ConvBNAct(inter_channel, inter_channel, c.kernel, c.stride, inter_channel, c.norm_layer, c.act)))
            block.append(('se', SEUnit(inter_channel, 4 * c.expand_ratio)))
            block.append(('point_wise', ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))

        self.block = nn.Sequential(OrderedDict(block))
        self.use_skip_connection = c.stride == 1 and c.in_ch == c.out_ch
        self.stochastic_path = StochasticDepth(sd_prob, "row")

    def forward(self, x):
        out = self.block(x)
        if self.use_skip_connection:
            out = x + self.stochastic_path(out)
        return out

# Pytorch Implementation of EfficientNetV2
class EfficientNetV2(nn.Module):
    
    def __init__(self, layer_infos, out_channels=1280, num_class = 0, dropout=0.2, stochastic_depth=0.0,
                 block=MBConv, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super(EfficientNetV2, self).__init__()
        self.layer_infos = layer_infos
        self.norm_layer = norm_layer
        self.act = act_layer

        self.in_channel = layer_infos[0].in_ch
        self.final_stage_channel = layer_infos[-1].out_ch
        self.out_channels = out_channels

        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in layer_infos)
        self.stochastic_depth = stochastic_depth

        self.stem = ConvBNAct(3, self.in_channel, 3, 2, 1, self.norm_layer, self.act)
        self.blocks = nn.Sequential(*self.make_stages(layer_infos, block))
        self.head = nn.Sequential(OrderedDict([
            ('bottleneck', ConvBNAct(self.final_stage_channel, out_channels, 1, 1, 1, self.norm_layer, self.act)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten()),
            ('dropout', nn.Dropout(p=dropout, inplace=True)),
            ('classifier', nn.Linear(out_channels, num_class) if num_class else nn.Identity())
        ]))

    def make_stages(self, layer_infos, block):
        return [layer for layer_info in layer_infos for layer in self.make_layers(copy.copy(layer_info), block)]

    def make_layers(self, layer_info, block):
        layers = []
        for i in range(layer_info.num_layers):
            layers.append(block(layer_info, sd_prob=self.get_sd_prob()))
            layer_info.in_ch = layer_info.out_ch
            layer_info.stride = 1
        return layers

    def get_sd_prob(self):
        sd_prob = self.stochastic_depth * (self.cur_block / self.num_block)
        self.cur_block += 1
        return sd_prob

    def forward(self, x):
      # print(x.shape)
      out_1 = self.stem(x)
      # print(out_1.shape)
      out_2 = self.blocks(out_1)
      # print(out_2.shape)
      out = self.head(out_2)
      # print(out.shape)
      return out

    def change_dropout_rate(self, p):
        self.head[-2] = nn.Dropout(p=p, inplace=True)

def efficientnet_v2_init(model):
  for m in model.modules():
      if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)

def get_efficientnet_v2_hyperparam(model_name):
    from box import Box
    # train_size, eval_size, dropout, randaug, mixup
    if 'efficientnet_v2_s' in model_name:
        end = 300, 384, 0.2, 10, 0
    elif 'efficientnet_v2_m' in model_name:
        end = 384, 480, 0.3, 15, 0.2
    elif 'efficientnet_v2_l' in model_name:
        end = 384, 480, 0.4, 20, 0.5
    return Box({"init_train_size": 128, "init_dropout": 0.1, "init_randaug": 5, "init_mixup": 0,
             "end_train_size": end[0], "end_dropout": end[2], "end_randaug": end[3], "end_mixup": end[4], "eval_size": end[1]})


def get_efficientnet_v2_structure(model_name):
    if 'efficientnet_v2_s' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 24, 24, 2, False, True),
            (4, 3, 2, 24, 48, 4, False, True),
            (4, 3, 2, 48, 64, 4, False, True),
            (4, 3, 2, 64, 128, 6, True, False),
            (6, 3, 1, 128, 160, 9, True, False),
            (6, 3, 2, 160, 256, 15, True, False),
        ]


num_class = 100
def efficientnet_v2(model_name, num_class):
    residual_config = [MBConvConfig(*layer_config) for layer_config in get_efficientnet_v2_structure(model_name)]
    model = EfficientNetV2(residual_config, 1280, num_class, dropout=0.1, stochastic_depth=0.2, block=MBConv, act_layer=nn.SiLU)
    efficientnet_v2_init(model)
    # load_from_zoo(model,model_name)  #pretrain model load
    return model

def efficientnet_v2_s(num_class): 
  return efficientnet_v2('efficientnet_v2_s', num_class)
def efficientnet_v2_m(num_class):
  return efficientnet_v2('efficientnet_v2_m', num_class)
def efficientnet_v2_l(num_class):
  return efficientnet_v2('efficientnet_v2_l', num_class)
