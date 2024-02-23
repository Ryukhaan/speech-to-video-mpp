import torch
from torch import nn
import torchvision
import numpy
import torch.nn.functional as F
from models.syncnet import SyncNet_color

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
        y_pred = y_pred[:, :, :, y_pred.size(3) // 2:]
        y_pred = torch.cat([y_pred[:, :, i] for i in range(self.number_of_frames)], dim=1)
        #audio = audio
        #y_pred = y_pred
        audio_emb, video_emb = self.net(audio, y_pred)
        #video_emb = video_emb.view(video_emb.size(0), 512, 4)
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

class LNetLoss(torch.nn.Module):
    def __init__(self):
        super(LNetLoss, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.lip_sync_loss = LipSyncLoss(device=self.device)
        self.lip_sync_loss.load_network("./checkpoints/lipsync_expert.pth")

    def forward(self, face_pred, face_true, audio_seq, T=5):
        lambda_1 = .5
        lambda_p = 1.
        lambda_sync = 0.3
        B, T, C, Hin, Win = face_pred.shape
        H, W = Hin, Win
        resizer = torchvision.transforms.Resize((H, W))
        y_pred = torch.zeros((B, T, C, H, W))
        y_true = torch.zeros((B, C, T, H, W))

        for i in range(T):
            y_pred[:,i,:,:,:] = resizer(face_pred[:,i, : ,:, :])
            y_true[:, :, i, :, :] = resizer(face_true[:, :, i, :, :])
        L1 = torch.nn.L1Loss()
        L_perceptual = VGGPerceptualLoss()
        l1_ = 0.
        lp_ = 0.
        for i in range(T):
            l1_ += L1(y_pred[:,i,:,:,:], y_true[:,:,i,:,:])
            lp_ += L_perceptual(y_pred[:,i,:,:,:], y_true[:,:,i,:,:])
        l1_val = l1_ / T
        lp_val = lp_ / T


        # L_sync = 0.0
        lsync_val = self.lip_sync_loss(audio_seq, y_pred.view(-1, T*C, H, W), y_true)
        #print(l1_val, lp_val, lsync_val)
        return lambda_1 * l1_val + lambda_p * lp_val + lambda_sync * lsync_val

class LoraLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.lip_sync_loss = LipSyncLoss(device=self.device)
        self.lip_sync_loss.load_network("./checkpoints/lipsync_expert.pth")

        self.L1 = torch.nn.SmoothL1Loss()
        self.L_perceptual = PerceptualLoss(device) #VGGPerceptualLoss()
        self.lambda_1 = 1.
        self.lambda_p = 1.
        self.lambda_sync = 0.3

    def forward(self, face_pred, face_true, audio_seq):

        B, C, T, Hin, Win = face_pred.shape
        _, _, _, H, W =  face_true.shape
        resizer = torchvision.transforms.Resize((Hin, Win),
                                                interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

        y_pred = torch.cat([face_pred[:, :, i] for i in range(face_pred.size(2))], dim=0)
        y_true = torch.cat([face_pred[:, :, i] for i in range(face_pred.size(2))], dim=0)

        #y_pred = resizer(y_pred)
        y_true  = resizer(y_true)

        print(y_pred.shape, y_true.shape)

        l1_val = self.L1(y_pred, y_true).to(self.device)
        #lp_val = self.L_perceptual(y_pred, y_true).to(self.device)
        lsync_val = self.lip_sync_loss(audio_seq, face_pred).to(self.device)

        return self.lambda_1 * l1_val #+ self.lambda_sync * lsync_val #+ self.lambda_p * lp_val
