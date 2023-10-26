import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from futils import flow_util
from models.base_blocks import LayerNorm2d, ADAINHourglass, FineEncoder, FineDecoder

class PNet(nn.Module):

    def __init__(self, coef_nc=19, descriptor_nc=256, nlayer=3):
        super(PNet, self).__init__()

        self.nlayer = nlayer
        nonlineartiry = nn.LeakyReLU(0.1)

        self.first = nn.Sequential(
            torch.nn.Conv1d(coef_nc, descriptor_nc, kernel_size=7, padding=0, bias=True)
        )

        for i in range(nlayer):
            net = nn.Sequential(nonlineartiry,
                                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3)
                                )
            setattr(self, 'encoder' + str(i), net)

        self.pooling = nn.AdativeAvgPool1d(1)
        self.output_nc = descriptor_nc
    def forward(self, phoneme):
        out = self.first(phoneme)
        for i in range(self.nlayer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:,:,3:-3]
        out = self.pooling(out)
        return out