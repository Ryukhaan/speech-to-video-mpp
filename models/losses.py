import torch
from torch import nn
import torchvision
import numpy
import torch.nn.functional as F
from models.syncnet import SyncNet_color

from torch import Tensor
from spectrum import get_spectrum

class PerceptualLoss(torch.nn.Module):
    def __init__(self, device, conv_index='22'):
        super(PerceptualLoss, self).__init__()
        self.vgg_layers = torchvision.models.vgg19().features
        modules = [m for m in self.vgg_layers]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8]).to(device)
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35]).to(device)

        #vgg_mean = (0.485, 0.456, 0.406)
        #vgg_std = (0.229, 0.224, 0.225)
        #self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute VGG/Perceptual loss between Super-Resolved and High-Resolution

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted model output tensor
        y_true : torch.Tensor
            Original image tensor

        Returns
        -------
        loss : torch.Tensor
            Perceptual VGG loss between sr and hr

        """

        def _forward(x):
            # x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(y_pred)
        with torch.no_grad():
            vgg_hr = _forward(y_true.detach())
        loss = F.mse_loss(vgg_sr, vgg_hr)
        return loss

class MSESpectrumLoss(torch.nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super(MSESpectrumLoss, self).__init__(*args, **kwargs)

    @staticmethod
    def get_log_spectrum(input):
        spectra = get_spectrum(input.flatten(0, 1)).unflatten(0, input.shape[:2])
        spectra = spectra.mean(dim=1)             # average over channels
        return (1 + spectra).log()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_spectrum = self.get_log_spectrum(input)
        target_spectrum = self.get_log_spectrum(target)
        return super(MSESpectrumLoss, self).forward(input_spectrum, target_spectrum)