import torch.nn as nn
from torchvision.models import vgg16

def custom_vgg():
    """Function creates a custom vgg16 model for binary classification"""
    vgg16_freeze = vgg16(weights="DEFAULT").features

    for param in vgg16_freeze.parameters():
        param.requires_grad = False

    model = nn.Sequential(
        vgg16_freeze,
        nn.Flatten(), 
        nn.Linear(8192, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return model


