import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from collections import OrderedDict
import numpy as np

class DorPatchClassifier(torch.nn.Module):
    def __init__(self):
        super(DorPatchClassifier, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        #feed in data x and get an output representing prediction
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x