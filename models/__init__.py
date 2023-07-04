import torch
from models.DNet import DNet
from models.LNet import LNet
from models.ENet import ENet

from torchsummary import summary

def _load(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

def load_checkpoint(path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"] if 'arcface' not in path else checkpoint
    new_s = {}
    for k, v in s.items():
        if 'low_res' in k:
            continue
        else:
            new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s, strict=False)
    return model

def load_network(args):
    torch.cuda.empty_cache()
    L_net = LNet()
    L_net = load_checkpoint(args.LNet_path, L_net)
    E_net = ENet(lnet=L_net)
    model = load_checkpoint(args.ENet_path, E_net)
    return model.eval()

def load_training_networks(args):
    torch.cuda.empty_cache()
    D_Net = DNet()
    print("Load checkpoint from: {}".format(args.DNet_path))
    checkpoint =  torch.load(args.DNet_path, map_location=lambda storage, loc: storage)
    D_Net.load_state_dict(checkpoint['net_G_ema'], strict=False)
    L_net = LNet()
    #summary(L_net, [(1, 80, 16), (6, 384, 384)])
    L_net = load_checkpoint(args.LNet_path, L_net)
    E_net = ENet(lnet=L_net)
    model = load_checkpoint(args.ENet_path, E_net)
    return D_Net, L_net, model

def load_DNet(args):
    torch.cuda.empty_cache()
    D_Net = DNet()
    print("Load checkpoint from: {}".format(args.DNet_path))
    checkpoint =  torch.load(args.DNet_path, map_location=lambda storage, loc: storage)
    D_Net.load_state_dict(checkpoint['net_G_ema'], strict=False)
    return D_Net.eval()