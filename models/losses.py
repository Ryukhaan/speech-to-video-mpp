import torch
from torch import nn
import torchvision
import numpy
import torch.nn.functional as F
from torch import optim

from models.syncnet import SyncNet_color
from models.ssim import SSIM, MS_SSIM

class LipSyncLoss(torch.nn.Module):
    def __init__(self, device):
        super(LipSyncLoss, self).__init__()
        self.device = device
        self.l2_loss = torch.nn.MSELoss()
        self.net = None
        self.number_of_frames = 5
        self.p_sync = torch.nn.CosineSimilarity()
        self.log_loss = nn.BCELoss()

    def load_network(self, path):
        self.net = SyncNet_color()
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint["state_dict"])
        for param in self.net.parameters():
            param.requires_grad = False
        self.net = self.net.to(self.device)


    def cosine_loss(self, a, v, y):
        d = nn.functional.cosine_similarity(a, v)
        loss = self.log_loss(d.unsqueeze(1), y)
        return loss

    def forward(self, audio, y_pred):
        y_pred = y_pred[:,:,y_pred.size(2)//2:]
        audio_emb, video_emb = self.net(audio, y_pred)
        y = torch.ones(y_pred.size(0), 1).float().to(self.device)
        p = self.cosine_loss(audio_emb, video_emb, y)
        return p


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

class TotalVariationLoss(torch.nn.Module):
     def __init__(self, device):
         super(TotalVariationLoss, self).__init__()
         self.device = device

     def forward(self, y_pred):
         """Compute total variation statistics on current batch."""
         diff1 = y_pred[..., 1:, :] - y_pred[..., :-1, :]
         diff2 = y_pred[..., :, 1:] - y_pred[..., :, :-1]

         res1 = diff1.abs().sum([1, 2, 3])
         res2 = diff2.abs().sum([1, 2, 3])
         #score = res1 + res2
         return torch.add(res1, res2).mean()

class LoraLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.lip_sync_loss = LipSyncLoss(device=self.device)
        self.lip_sync_loss.load_network("./checkpoints/lipsync_expert.pth")

        self.L1 = torch.nn.L1Loss()
        self.L_perceptual = PerceptualLoss(device) #VGGPerceptualLoss()
        self.tv_loss = TotalVariationLoss(device)
        self.ssim_loss = MS_SSIM(data_range=1.0)
        self.lambda_1 = 1.
        self.lambda_p = 1.
        self.lambda_sync = .01
        #self.lambda_tv = 0.1
        #self.lambda_ssim = 0.84

    def forward(self, face_pred, face_true, audio_seq):
        B = audio_seq.size(0)
        input_dim_size = len(face_pred.size())
        Hin, Win = face_pred.shape[-2:]
        H, W =  face_true.shape[-2:]

        resizer_96 = torchvision.transforms.Resize((96, 96))
        resizer_up = torchvision.transforms.Resize((H, W))

        if input_dim_size > 4:
            y_pred = torch.cat([face_pred[:, :, i] for i in range(face_pred.size(2))], dim=0)
            face_seq = torch.cat([face_pred[:, :, i] for i in range(face_pred.size(2))], dim=1)
            y_true = torch.cat([face_true[:, :, i] for i in range(face_pred.size(2))], dim=0)
            #audio_cat = torch.cat([audio_seq[:, i] for i in range(face_pred.size(2))], dim=0)
        else:
            y_pred = face_pred
            y_true = face_true

        y_pred_up = resizer_up(y_pred)
        face_seq = resizer_96(face_seq)
        y_pred_96 = resizer_96(y_pred)
        y_true_96 = resizer_96(y_true)

        l1_val = self.L1(y_pred_up, y_true).to(self.device)
        lssim_val = self.ssim_loss(y_pred_up, y_true).to(self.device)
        lp_val = self.L_perceptual(y_pred_96, y_true_96).to(self.device)
        lsync_val = self.lip_sync_loss(audio_seq, face_seq).to(self.device)
        tv_val = self.tv_loss(y_pred_up).to(self.device)
        return self.lambda_1 * l1_val \
            + self.lambda_sync * lsync_val \
            + self.lambda_p * lp_val \
            #+ self.lambda_ssim * lssim_val \
            #+ self.lambda_tv * tv_val
