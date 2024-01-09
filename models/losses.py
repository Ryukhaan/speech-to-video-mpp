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
    def cosine_loss(self, a, v, y):
        d = nn.functional.cosine_similarity(a, v)
        loss = self.log_loss(d.unsqueeze(1), y)
        return loss

    def forward(self, audio, y_pred, y_true):
        audio_emb, video_emb = self.net(audio, y_pred)
        p = self.cosine_loss(video_emb, audio_emb, y_true)
        return torch.nn.mean(-torch.log(p))

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


        resizer = torchvision.transforms.Resize((384, 384))
        y_pred = torch.zeros(face_pred.shape[:-2]+(384,384))
        y_true = torch.zeros(face_true.shape[:-2]+(384,384))
        for i in range(T):
        #    print( resizer(face_pred[:,i, : ,:, :]).shape)
            y_pred[:,i,:,:,:] = resizer(face_pred[:,i, : ,:, :])
            y_true[:, i, :, :, :] = resizer(face_true[:, :, i, :, :])
        #y_pred = torch.cat(y_pred, dim=1)
        #y_true = torch.cat(y_true, dim=2)
        L1 = torch.nn.L1Loss()
        L_perceptual = VGGPerceptualLoss()
        l1_ = []
        lp_ = []
        for i in range(T):
            print("Pred", y_pred.shape, y_true.shape)
            l1_.append(L1(y_pred[:,i,:,:,:], y_true[:,i,:,:,:,:]))
            lp_.append(L_perceptual(y_pred[:,i,:,:,:], y_true[:,i,:,:,:]))
        l1_val = torch.sum(l1_)
        lp_val = torch.sum(lp_)
        #lp_val = L_perceptual(y_pred, y_true)

        # L_sync = 0.0
        lsync_val = self.lip_sync_loss(audio_seq, y_pred, y_true)
        print(l1_val, lp_val, lsync_val)
        return lambda_1 * l1_val + lambda_p * lp_val + lambda_sync * lsync_val
