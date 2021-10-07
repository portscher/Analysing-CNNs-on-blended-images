import os

import torch

from .model import Model
from architectures.efficientnet.efficient_definition2 import EfficientNet


class EfficientNetB0(Model):

    def __init__(self, train_from_scratch=True, path=None, attention='none'):
        super().__init__(path, train_from_scratch, attention)
        self.path = path
        self.train_from_scratch = train_from_scratch
        self.attention = attention

    def get_model(self):
        if self.attention == 'aacn':
            model = EfficientNet(num_classes=8, attention=True)
        elif self.attention == 'cbam':
            raise NotImplementedError('CBAM not yet implemented for EfficientNet')
        else:
            model = EfficientNet(num_classes=8, attention=False)

        if not self.train_from_scratch and os.path.isfile(self.path):
            print("Loading efficientnet from disk")
            model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu'))['state_dict'])
            model.eval()

        return model
