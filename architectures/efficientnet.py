import os

import torch

from .efficientnet_definition import EfficientNet
from .model import Model


class EfficientNetB0(Model):

    def __init__(self, train_from_scratch=True, path=None):
        super().__init__(path, train_from_scratch)
        self.path = path
        self.train_from_scratch = train_from_scratch

    def get_model(self):
        model = EfficientNet.from_name('efficientnet-b0', num_classes=8)

        if not self.train_from_scratch and os.path.isfile(self.path):
            print("Loading efficientnet from disk")
            model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu'))['state_dict'])
            model.eval()

        return model
