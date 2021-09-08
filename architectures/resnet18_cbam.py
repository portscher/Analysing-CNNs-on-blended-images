import os

import torch

from .model import Model
from .resnet18_cbam_definition import ResidualNet


class ResNet18CBAM(Model):

    def __init__(self, train_from_scratch=True, path=None):
        super().__init__(path, train_from_scratch)
        self.path = path
        self.train_from_scratch = train_from_scratch

    def get_model(self):
        model = ResidualNet(18, 8, att_type=None)

        if not self.train_from_scratch and os.path.isfile(self.path):
            print("Loading resnet18 with CBAM from disk")
            model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu'))['state_dict'])
            model.eval()

        return model
