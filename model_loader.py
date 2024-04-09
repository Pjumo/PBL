import models.resnet as resnet
import models.cnn as cnn
import models.u2net as u2net  # µ2Net 모델을 불러오기 위한 import 문 추가

models = {
    'cnn18': cnn.CNN18,
    'cnn34': cnn.CNN34,
    'resnet18': resnet.ResNet18,
    'resnet34': resnet.ResNet34,
    'u2net': u2net.u2net_caller  # µ2Net 모델 추가
}


def load(model_name, num_class):
    net = models[model_name](num_classes=num_class)
    return net
