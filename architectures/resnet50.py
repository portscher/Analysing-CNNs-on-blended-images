import os

import torch
import torch.nn as nn
from torchvision import models as models

from .aacn import AACN_ResNet
from .model import Model


class ResNet50(Model):

    def __init__(self, train_from_scratch=True, path=None, attention='none'):
        super().__init__(path, train_from_scratch, attention)
        self.path = path
        self.train_from_scratch = train_from_scratch
        self.attention = attention

    def get_model(self):
        if self.attention == 'cbam':
            raise NotImplementedError('CBAM not yet implemented for ResNet50')
        elif self.attention == 'aacn':
            model = AACN_ResNet.resnet50(num_classes=8, attention=[False, True, True, True], num_heads=4, k=2, v=0.25, image_size=224)
        else:
            model = models.resnet50(progress=True)
            # adjust the classification layer to classify 8 object types
            model.fc = nn.Linear(2048, 8)

        if not self.train_from_scratch and os.path.isfile(self.path):
            print("Loading resnet50 from disk")
            model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu'))['state_dict'])
            model.eval()

        return model
