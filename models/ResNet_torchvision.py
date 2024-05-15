# resnet: torchvision version

import torch
import torch.nn as nn
import torchvision.models as models

def create_model(m_type='resnet101',num_classes=1000, pretrained = False):
    # create various resnet models
    if m_type == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif m_type == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif m_type == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    elif m_type == 'resnext50':
        model = models.resnext50_32x4d(pretrained=pretrained)
    elif m_type == 'resnext101':
        model = models.resnext101_32x8d(pretrained=pretrained)
    else:
        raise ValueError('Wrong Model Type')
        
    model = ResNet(model, num_classes)
    return model

class ResNet(nn.Module):
    def __init__(self, model, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = nn.Linear(model.fc.in_features, num_classes)
        # self.model.fc = nn.Identity()
    def forward(self, x, feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)        
        x = self.layer1(x)        
        x = self.layer2(x)        
        x = self.layer3(x)        
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        x = self.fc(feat)
        if feature:
            return x, feat
        else:
            return x

    def feat_nograd_forward(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)        
            x = self.layer1(x)        
            x = self.layer2(x)        
            x = self.layer3(x)        
            x = self.layer4(x)
            x = self.avgpool(x)
            feat = torch.flatten(x, 1)
        x = self.fc(feat)
        return x, feat