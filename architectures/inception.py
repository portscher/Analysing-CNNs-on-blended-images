import os

import torch
import torch.nn as nn
from torchvision import models as models

from architectures.model import Model


class Inception(Model):

    def __init__(self, train_from_scratch=True, path=None):

        super().__init__(path, train_from_scratch)
        self.train_from_scratch = train_from_scratch

    def get_model(self):
        model = models.inception_v3(progress=True, init_weights=True)

        # adjust the classification layer to classify 8 object types
        model.fc = nn.Linear(2048, 8)

        if not self.train_from_scratch and os.path.isfile(self.path):
            print("Loading inception from disk")
            model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu'))['state_dict'])
            model.eval()

        return model
