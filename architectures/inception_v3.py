import os

import torch
import torch.nn as nn

from architectures.inception import inception_definition
from architectures.model import Model


class Inception(Model):

    def __init__(self, train_from_scratch=True, path=None, attention='none', heads=4):

        super().__init__(path, train_from_scratch, attention, heads)
        self.train_from_scratch = train_from_scratch
        self.path = path
        self.attention = attention
        self.heads = heads

    def get_model(self):
        if self.attention == 'aacn':
            model = inception_definition.inception_v3(pretrained=False, progress=True, attention='aacn')
        else:
            model = inception_definition.inception_v3(pretrained=False, progress=True, attention='none')

        # adjust the classification layer to classify 8 object types
        model.fc = nn.Linear(2048, 8)

        if not self.train_from_scratch and os.path.isfile(self.path):
            print("Loading inception from disk")
            model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu'))['state_dict'])
            model.eval()

        return model
