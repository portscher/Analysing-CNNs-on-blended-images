import os

import torch

from .model import Model
from .aacn import AACN_ResNet
from torchvision import models as models
import torch.nn as nn


class ResNet18(Model):

    def __init__(self, train_from_scratch=True, path=None, attention='none', heads=4):
        super().__init__(path, train_from_scratch, attention, heads)
        self.path = path
        self.train_from_scratch = train_from_scratch
        self.attention = attention
        self.heads = heads

    def get_model(self):
        if self.attention == 'none':
            model = models.resnet18(progress=True)
            # adjust the classification layer to classify 8 object types
            model.fc = nn.Linear(512, 8)
        elif self.attention == 'aacn':
            model = AACN_ResNet.resnet18(num_classes=8, attention=[False, True, True, True], num_heads=self.heads, k=2, v=0.25, image_size=224)

        if not self.train_from_scratch and os.path.isfile(self.path):
            print("Loading resnet18 from disk")
            model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu'))['state_dict'])
            model.eval()

        return model
