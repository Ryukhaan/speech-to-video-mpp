import functools
import torch
import torch.nn as nn

from models.transformer import RETURNX, Transformer
from models.base_blocks import Conv2d, LayerNorm2d, FirstBlock2d, DownBlock2d, UpBlock2d, \
                               FFCADAINResBlocks, Jump, FinalBlock2d

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Visual_Encoder(nn.Module):
    def __init__(self, image_nc, ngf, img_f, layers, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(Visual_Encoder, self).__init__()
        self.layers = layers
        self.first_inp = FirstBlock2d(image_nc, ngf, norm_layer, nonlinearity, use_spect)
        self.first_ref = FirstBlock2d(image_nc, ngf, norm_layer, nonlinearity, use_spect)
        for i in range(layers):
            in_channels = min(ngf*(2**i), img_f)
            out_channels = min(ngf*(2**(i+1)), img_f)
            model_ref = DownBlock2d(in_channels, out_channels, norm_layer, nonlinearity, use_spect)
            model_inp = DownBlock2d(in_channels, out_channels, norm_layer, nonlinearity, use_spect)
            if i < 2:
                ca_layer = RETURNX()
            else:
                ca_layer = Transformer(2**(i+1) * ngf,2,4,ngf,ngf*4)
            setattr(self, 'ca' + str(i), ca_layer)
            setattr(self, 'ref_down' + str(i), model_ref)
            setattr(self, 'inp_down' + str(i), model_inp)
        self.output_nc = out_channels * 2

    def forward(self, maskGT, ref):
        x_maskGT, x_ref = self.first_inp(maskGT), self.first_ref(ref)
        out=[x_maskGT]
        for i in range(self.layers):
            model_ref = getattr(self, 'ref_down'+str(i))
            model_inp = getattr(self, 'inp_down'+str(i))
            ca_layer = getattr(self, 'ca'+str(i))
            x_maskGT, x_ref = model_inp(x_maskGT), model_ref(x_ref)
            x_maskGT = ca_layer(x_maskGT, x_ref)
            if i < self.layers - 1:
                out.append(x_maskGT)
            else:
                out.append(torch.cat([x_maskGT, x_ref], dim=1)) # concat ref features !
        return out


class Decoder(nn.Module):
    def __init__(self, image_nc, feature_nc, ngf, img_f, layers, num_block, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(Decoder, self).__init__()
        self.layers = layers
        for i in range(layers)[::-1]:
            if  i == layers-1:
                in_channels = ngf*(2**(i+1)) * 2
            else:
                in_channels = min(ngf*(2**(i+1)), img_f)
            out_channels = min(ngf*(2**i), img_f)
            up = UpBlock2d(in_channels, out_channels, norm_layer, nonlinearity, use_spect)
            res = FFCADAINResBlocks(num_block, in_channels, feature_nc, norm_layer, nonlinearity, use_spect)
            jump = Jump(out_channels, norm_layer, nonlinearity, use_spect)

            setattr(self, 'up' + str(i), up)
            setattr(self, 'res' + str(i), res)
            setattr(self, 'jump' + str(i), jump)

        self.final = FinalBlock2d(out_channels, image_nc, use_spect, 'sigmoid')
        self.output_nc = out_channels

    def forward(self, x, z):
        out = x.pop()
        for i in range(self.layers)[::-1]:
            res_model = getattr(self, 'res' + str(i))
            up_model = getattr(self, 'up' + str(i))
            jump_model = getattr(self, 'jump' + str(i))
            out = res_model(out, z)
            out = up_model(out)
            out = jump_model(x.pop()) + out
        out_image = self.final(out)
        return out_image


class AudioTextSyncNet(nn.Module):
    def __init__(
        self,
        image_nc=3,
        descriptor_nc=992,
        layer=3,
        base_nc=64,
        max_nc=512,
        num_res_blocks=9,
        use_spect=True,
        encoder=Visual_Encoder,
        decoder=Decoder
        ):
        super(AudioTextSyncNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True)
        kwargs = {'norm_layer':norm_layer, 'nonlinearity':nonlinearity, 'use_spect':use_spect}
        self.descriptor_nc = descriptor_nc

        self.encoder = encoder(image_nc, base_nc, max_nc, layer, **kwargs)
        self.decoder = decoder(image_nc, self.descriptor_nc, base_nc, max_nc, layer, num_res_blocks, **kwargs)

    def forward(self, audio_sequences, text_sequences, face_sequences):
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:,i] for i in range(audio_sequences.size(1))], dim=0)
            text_sequences = torch.cat([text_sequences[:,i] for i in range(text_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        cropped, ref = torch.split(face_sequences, 3, dim=1)

        z_features = torch.cat([audio_sequences, text_sequences], dim=1)

        vis_feat = self.encoder(cropped, ref)
        _outputs = self.decoder(vis_feat, z_features)

        if input_dim_size > 4:
            _outputs = torch.split(_outputs, B, dim=0)
            outputs = torch.stack(_outputs, dim=2)
        else:
            outputs = _outputs
        return outputs


class Wav2Lip_disc_qual(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)),  # 48,96

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),  # 48,48
                          nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 24,24
                          nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # 12,12
                          nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 6,6
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1), ),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                          nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1),
                                                 torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)
