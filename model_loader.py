import models.resnet as resnet
import models.cnn as cnn
import models.u2net as u2net  # µ2Net 모델을 불러오기 위한 import 문 추가

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
    'u2net': u2net.u2net_caller  # µ2Net 모델 추가
}


def load(model_name, num_class):
    net = models[model_name](num_classes=num_class)
    return net
