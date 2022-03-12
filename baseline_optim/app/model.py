import torch
from collections import OrderedDict
import torchvision.models as models
import torch.nn as nn

def base_model(pretrained=False):

    model = models.resnet18(pretrained=pretrained)
    classifier=nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(model.fc.in_features, 100)),
        ('added_relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(100, 50)),
        ('added_relu2', nn.ReLU(inplace=True)),
        ('fc3', nn.Linear(50, 25))
        ]))
    model.fc=classifier

    return model
