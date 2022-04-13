import torch
from collections import OrderedDict
import torchvision.models as models
import torch.nn as nn
from inception_resnet_v1 import InceptionResnetV1
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class normalize(nn.Module):
    def __init__(self):
        super(normalize, self).__init__()
        
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x


def base_model(pretrained=False, arch='resnet18'):
    
    if arch=='resnet18':
        model = models.resnet18(pretrained=pretrained)
        classifier=nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(model.fc.in_features, 100)),
            ('added_relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(100, 50)),
            ('added_relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(50, 25))
            ]))
        model.fc=classifier
    
    elif arch=='resnet50':

        model = models.resnet50(pretrained=pretrained)
        classifier=nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(model.fc.in_features, 512)),
            ('added_relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(512, 128)),
            ('added_relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(128, 64))
            ]))
        model.fc=classifier
    
    elif arch=='resnet101':

        model = models.resnet101(pretrained=pretrained)
        classifier=nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(model.fc.in_features, 512)),
            ('added_relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(512, 128)),
            ('added_relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(128, 64))
            ]))
        model.fc=classifier
    
    elif arch=='inception':

        model = InceptionResnetV1()
        if pretrained:
            pretrained_dict = torch.load('./20180402-114759-vggface2.pt')
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(pretrained_dict)
        
        model = nn.Sequential(*list(model.children())[:-5])
        model.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
        model.last_linear = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=1792, out_features=512, bias=False),
            normalize())

        model.add_module('fc1', nn.Linear(512, 100))
        model.add_module('added_relu1', nn.ReLU(inplace=True))
        model.add_module('fc2', nn.Linear(100, 50))
        model.add_module('added_relu2', nn.ReLU(inplace=True))
        model.add_module('fc3', nn.Linear(50, 25))

        # Congelamos los 10 primeros bloques
        
        ct = 0
        for child in model.children():
            ct += 1
            if ct < 11:
                for param in child.parameters():
                    param.requires_grad = False
        
        
    elif arch=='vgg16':
        
        model=models.vgg16(pretrained=False)
        classifier=nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 100)),
            ('added_relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(100, 50)),
            ('added_relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(50, 25))
            ]))
        model.classifier=classifier

    return model



