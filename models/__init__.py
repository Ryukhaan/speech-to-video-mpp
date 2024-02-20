import torch
from models.DNet import DNet
from models.LNet import LNet
from models.ENet import ENet

from peft import LoraConfig, get_peft_model

from torchsummary import summary

global_step = 0
glbal_epoch = 0

def _load(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

def load_checkpoint_lipsync(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

def load_checkpoint(path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    try:
        s = checkpoint["state_dict"] if 'arcface' not in path else checkpoint
        new_s = {}
        for k, v in s.items():
            if 'low_res' in k:
                continue
            else:
                new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s, strict=False)
    except:
        model.load_state_dict(torch.load(path))
        #model = torch.load(path)
    return model

def load_network(args):
    torch.cuda.empty_cache()
    L_net = LNet()
    L_net = load_checkpoint(args.LNet_path, L_net)
    E_net = ENet(lnet=L_net)
    model = load_checkpoint(args.ENet_path, E_net)
    return model.eval()

def load_lora_network(args):
    torch.cuda.empty_cache()

    L_net = LNet()
    decoder_config = LoraConfig(
        r=8,
        lora_alpha=4,
        target_modules=["mlp_gamma", "mlp_beta", "mlp_shared.0"],
        lora_dropout=0.0,
        bias="none",
    )
    audio_enc_config = LoraConfig(
        r=2,
        lora_alpha=2,
        target_modules=["conv_block.0"],
        lora_dropout=0.0
    )
    lora_l_decoder = get_peft_model(L_net.decoder, decoder_config)
    lora_ae_encode = get_peft_model(L_net.audio_encoder, audio_enc_config)
    L_net.decoder = lora_l_decoder
    L_net.audio_encoder = lora_ae_encode
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