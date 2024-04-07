import models.resnet as resnet

models = {
    'resnet18': resnet.ResNet18,
    'resnet34': resnet.ResNet34
}


def load(model_name, num_class):
    net = models[model_name](num_classes=num_class)
    return net
