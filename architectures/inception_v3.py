import os

import torch
import torch.nn as nn

from architectures.incpetion import inception_definition
from architectures.model import Model


class Inception(Model):

    def __init__(self, train_from_scratch=True, path=None, attention='none'):

        super().__init__(path, train_from_scratch, attention)
        self.train_from_scratch = train_from_scratch
        self.path = path
        self.attention = attention

    def get_model(self):
        if self.attention == 'aacn':
            raise NotImplementedError('AACN not yet implemented for Inception')
        elif self.attention == 'cbam':
            raise NotImplementedError('CBAM not yet implemented for Inception')
        else:
            model = inception_definition.inception_v3(pretrained=False, progress=True)

        # adjust the classification layer to classify 8 object types
        model.fc = nn.Linear(2048, 8)

        if not self.train_from_scratch and os.path.isfile(self.path):
            print("Loading inception from disk")
            model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu'))['state_dict'])
            model.eval()

        return model
