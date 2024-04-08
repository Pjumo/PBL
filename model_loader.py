import models.resnet as resnet
import models.cnn as cnn

models = {
    'cnn18': cnn.CNN18,
    'cnn34': cnn.CNN34,
    'resnet18': resnet.ResNet18,
    'resnet34': resnet.ResNet34
}


def load(model_name, num_class):
    net = models[model_name](num_classes=num_class)
    return net
