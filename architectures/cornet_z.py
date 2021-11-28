import os
from collections import OrderedDict

import torch
import torch.utils.model_zoo
from torch import nn

from .aacn import AACN_Layer
from .model import Model


class CORnet(Model):

    def __init__(self, train_from_scratch=True, path=None, attention='none'):
        super().__init__(path, train_from_scratch, attention)
        self.path = path
        self.train_from_scratch = train_from_scratch
        self.attention = attention

    def get_model(self):
        if self.attention == 'aacn':
            model = torch.nn.DataParallel(CORnet_Z(attention=True))
        elif self.attention == 'cbam':
            raise NotImplementedError('CBAM not yet implemented for CORnet_Z')
        else:
            model = torch.nn.DataParallel(CORnet_Z(attention=False))

        if not self.train_from_scratch and os.path.isfile(self.path):
            print("Loading cornet-z from disk")
            model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu'))['state_dict'])
            model.eval()

        return model


# All code below this point:
# Authors: qbilius, mschrimpf (github username)
# Github repo: https://github.com/dicarlolab/CORnet

class Flatten(nn.Module):
    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_Z(nn.Module):

    def __init__(self, in_channels, out_channels, img_size, kernel_size=3, stride=1, att=False):
        super().__init__()
        if att is False:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                                  stride=(stride, stride), padding=kernel_size // 2)
        else:
            self.conv = AACN_Layer(in_channels=in_channels, out_channels=out_channels, dk=40, dv=4,
                                   kernel_size=kernel_size, num_heads=4, image_size=img_size, inference=False)

        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x


def CORnet_Z(attention):
    if attention:
        # replace all 3x3 convolutions with an attention-augmented convolution
        att = [False, True, True, True]
    else:
        att = [False, False, False, False]

    model = nn.Sequential(OrderedDict([
        ('V1', CORblock_Z(3, 64, kernel_size=7, stride=2, img_size=224, att=att[0])),
        ('V2', CORblock_Z(64, 128, img_size=224 // 4, att=att[1])),
        ('V4', CORblock_Z(128, 256, img_size=224 // 8, att=att[2])),
        ('IT', CORblock_Z(256, 512, img_size=224 // 16, att=att[3])),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 8)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model
