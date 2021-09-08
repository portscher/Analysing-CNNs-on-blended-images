import os

import torch

from .model import Model
from .aacn import AACN_Model


class ResNet18AACN(Model):

    def __init__(self, train_from_scratch=True, path=None):
        super().__init__(path, train_from_scratch)
        self.path = path
        self.train_from_scratch = train_from_scratch

    def get_model(self):
        # with aacn attention: attention=[False, True, True, True]
        model = AACN_Model.resnet18(num_classes=8, attention=[False, True, True, True], num_heads=4, k=2, v=0.25, image_size=224)

        if not self.train_from_scratch and os.path.isfile(self.path):
            print("Loading resnet18 with AACN from disk")
            model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu'))['state_dict'])
            model.eval()

        return model
