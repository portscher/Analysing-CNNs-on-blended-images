import os

import torch


class Model:

    def __init__(self, path, train_from_scratch):
        self.path = path
        self.train_from_scratch = train_from_scratch

    def get_model(self):
        """
        returns the model
        """
        pass

    def get_num_classes(self):
        """
        returns the amount of classes the model was trained on
        """
        if os.path.isfile(self.path):
            return torch.load(self.path, map_location=torch.device('cpu'))['num_classes']

    def get_num_train_imgs(self):
        """
        returns the amount of classes the training images the model was trained on
        """
        if os.path.isfile(self.path):
            return torch.load(self.path, map_location=torch.device('cpu'))['num_train_imgs']

    def get_learning_rate(self):
        """
        returns the learning rate used for training the model
        """
        if os.path.isfile(self.path):
            return torch.load(self.path, map_location=torch.device('cpu'))['epoch']

    def get_num_epochs(self):
        """
        returns number of epochs the model was trained for
        """
        if os.path.isfile(self.path):
            return torch.load(self.path, map_location=torch.device('cpu'))['learning_rate']

    def get_batch_size(self):
        """
        returns the batch size used for training the model
        """
        if os.path.isfile(self.path):
            return torch.load(self.path, map_location=torch.device('cpu'))['batch_size']
