import torch
from torch import nn
import torchvision
import numpy

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

class ArcFaceLoss(torch.nn.Module):
    def __init__(self, device):
        super(ArcFaceLoss, self).__init__()
        self.face3d_net_path = 'checkpoints/face3d_pretrain_epoch_20.pth'
        self.device = device
        self.lm3d = 'checkpoints/BFM'
        self.l2_loss = torch.nn.MSELoss()

    def forward(self, y_pred, y_true):

        torch.cuda.empty_cache()
        net_recon = load_face3d_net(self.face3d_net_path, self.device)
        lm3d_std = load_lm3d(self.lm3d)

        _, C, W, H = y_pred.shape

        lm_idx = lm[idx].reshape([-1, 2])
        if np.mean(lm_idx) == -1:
            lm_idx = (lm3d_std[:, :2]+1) / 2.
            lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
        else:
            lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

        # Y Predicted
        d_ypred = y_pred.cpu().detach().numpy()
        d_ypred = cv2.fromarray(d_ypred)
        trans_params, im_idx, lm_idx, _ = align_img(d_ypred[0], lm_idx, lm3d_std)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_idx_tensor = torch.tensor(np.array(im_idx)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
        with torch.no_grad():
            coeffs = split_coeff(net_recon(im_idx_tensor))
        pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
        pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'],\
                                         pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
        # Y Targets
        #d_ytrue = y_true.cpu().detach().numpy()
        #d_ytrue = cv2.fromarray(d_ytrue)
        #trans_params, im_idx, lm_idx, _ = align_img(d_ytrue[0], lm_idx, lm3d_std)
        #trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        #im_idx_tensor = torch.tensor(np.array(im_idx) / 255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
        #with torch.no_grad():
        #    coeffs = split_coeff(net_recon(im_idx_tensor))
        #true_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}
        #true_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'], \
        #                             pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
        return self.l2_loss(pred_coeff, y_true)

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval().to('cuda'))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval().to('cuda'))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval().to('cuda'))
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval().to('cuda'))
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to('cuda'))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to('cuda'))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        #input = (input-self.mean) / self.std
        #target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input.to('cuda')
        y = target.to('cuda')
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
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

        self.L1 = torch.nn.L1Loss()
        self.L_perceptual = VGGPerceptualLoss()
        self.lambda_1 = 1.
        self.lambda_p = 1.
        self.lambda_sync = 0.3

    def forward(self, face_pred, face_true, audio_seq):

        B, C, T, Hin, Win = face_pred.shape
        _, _, _, H, W =  face_true.shape
        resizer = torchvision.transforms.Resize((H, W))

        y_pred = face_pred.view(B*T, C, Hin, Win)
        y_true = face_true.view(B*T, C, H, W)

        y_pred = resizer(y_pred)
        y_true  = resizer(y_true)

        l1_val = self.L1(y_pred, y_true).to(self.device)
        lp_val = self.L_perceptual(y_pred, y_true).to(self.device)
        lsync_val = self.lip_sync_loss(audio_seq, face_pred).to(self.device)

        return self.lambda_1 * l1_val + self.lambda_p * lp_val + self.lambda_sync * lsync_val
