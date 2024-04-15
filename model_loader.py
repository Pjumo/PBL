import models.resnet as resnet
import models.cnn as cnn
import models.u2net as u2net  # µ2Net 모델을 불러오기 위한 import 문 추가
import models.effnet as effnet
import models.effnetv2 as effnetv2

models = {
    'cnn18': cnn.CNN18,
    'cnn34': cnn.CNN34,

    'resnet18': resnet.ResNet18,
    'resnet34': resnet.ResNet34,
    'resnet50': resnet.ResNet50,
    'resnet101': resnet.ResNet101,
    'resnet152': resnet.ResNet152,

    'preact_resnet18': resnet.PreActResNet18,
    'preact_resnet34': resnet.PreActResNet34,
    'preact_resnet50': resnet.PreActResNet50,

    'preact_resnet101': resnet.PreActResNet101,
    'preact_resnet152': resnet.PreActResNet152,

    'u2net': u2net.u2net_caller,

      # µ2Net 모델 추가

    'effnet_b0' : effnet.efficientnet_b0,
    'effnet_b1' : effnet.efficientnet_b1,
    'effnet_b2' : effnet.efficientnet_b2,
    'efficientnet_b3' : effnet.efficientnet_b3,
    'efficientnet_b4' : effnet.efficientnet_b4,
    'efficientnet_b5' : effnet.efficientnet_b5,
    'efficientnet_b6' : effnet.efficientnet_b6,

    'effnetv2_s' : effnetv2.efficientnet_v2_s,
    'effnetv2_m' : effnetv2.efficientnet_v2_m,
    'effnetv2_l' : effnetv2.efficientnet_v2_l,
}

def load(model_name, num_class):
    if model_name in models:
        net = models[model_name](num_class)
        return net
    else:
        raise ValueError("Model '{}' not found in the list of available models.".format(model_name))