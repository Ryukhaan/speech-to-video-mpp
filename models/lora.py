import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

from models.ffc import FFC
from basicsr.archs.arch_util import default_init_weights

from base_blocks import ADAIN

from peft import LoraConfig, get_peft_model

class LoraLayer():
    def __init__(self,
                 rank,
                 alpha,
                 dropout):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        else:
            self.dropout_layer = lambda x: x

class LoraADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc, rank, ada_module, use_bias=False):
        super().__init__()

        self.ada = ada_module

        self.shared_A = nn.Sequential(
            nn.Linear(bias=use_bias),
            nn.ReLU()
        )

        self.gamma_A = nn.Linear(bias=use_bias)
        self.gamma_B = nn.Linear(bias=use_bias)

        self.beta_A = nn.Linear(bias=use_bias)
        self.beta_B = nn.Linear(bias=use_bias)

    def forward(self, x, feature):
