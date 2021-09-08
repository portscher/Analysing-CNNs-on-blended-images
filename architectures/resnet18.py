import os

import torch
import torch.nn as nn
from torchvision import models as models

from .model import Model


class ResNet18(Model):

    def __init__(self, train_from_scratch=True, path=None):
        super().__init__(path, train_from_scratch)
        self.path = path
        self.train_from_scratch = train_from_scratch

    def get_model(self):
        model = models.resnet18(progress=True)

        # adjust the classification layer to classify 8 object types
        model.fc = nn.Linear(512, 8)

        if not self.train_from_scratch and os.path.isfile(self.path):
            print("Loading resnet18 from disk")
            model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu'))['state_dict'])
            model.eval()

        return model
