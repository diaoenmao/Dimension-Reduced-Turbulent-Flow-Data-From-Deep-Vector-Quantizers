import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from modules import VectorQuantization3d
from .utils import init_param

Normalization = nn.BatchNorm3d
Activation = nn.ReLU
Conv = nn.Conv3d


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            Normalization(in_channel),
            Activation(inplace=True),
            Conv(in_channel, channel, 3, 1, 1),
            Normalization(channel),
            Activation(inplace=True),
            Conv(channel, in_channel, 1, 1, 0),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, num_res_block, num_res_channel, stride):
        super().__init__()
        if stride == 4:
            blocks = [
                Conv(in_channel, channel // 2, 4, 2, 1),
                Normalization(channel // 2),
                Activation(inplace=True),
                Conv(channel // 2, channel, 4, 2, 1),
                Normalization(channel),
                Activation(inplace=True),
                Conv(channel, channel, 3, 1, 1),
            ]
        elif stride == 2:
            blocks = [
                Conv(in_channel, channel // 2, 4, 2, 1),
                Normalization(channel // 2),
                Activation(inplace=True),
                Conv(channel // 2, channel, 3, 1, 1),
            ]
        else:
            raise ValueError('Not valid stride')
        for i in range(num_res_block):
            blocks.append(ResBlock(channel, num_res_channel))
        blocks.extend([
            Normalization(channel),
            Activation(inplace=True)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, num_res_block, num_res_channel, stride):
        super().__init__()
        blocks = [Conv(in_channel, channel, 3, 1, 1)]
        for i in range(num_res_block):
            blocks.append(ResBlock(channel, num_res_channel))
        blocks.extend([
            Normalization(channel),
            Activation(inplace=True)])
        if stride == 4:
            blocks.extend([
                nn.ConvTranspose3d(channel, channel // 2, 4, 2, 1),
                Normalization(channel // 2),
                Activation(inplace=True),
                nn.ConvTranspose3d(channel // 2, out_channel, 4, 2, 1),
            ])
        elif stride == 2:
            blocks.append(
                nn.ConvTranspose3d(channel, out_channel, 4, 2, 1)
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(self, in_channel=3, channel=128, num_res_block=2, num_res_channel=32, embedding_dim=64,
                 num_embedding=512, vq_commit=0.25):
        super().__init__()
        self.vq_commit = vq_commit
        self.encoder_bottom = Encoder(in_channel, channel, num_res_block, num_res_channel, stride=4)
        self.encoder_conv_bottom = Conv(embedding_dim + channel, embedding_dim, 1, 1, 0)
        self.quantizer_bottom = VectorQuantization3d(embedding_dim, num_embedding)
        self.decoder_bottom = Decoder(embedding_dim + embedding_dim, in_channel, channel, num_res_block,
                                      num_res_channel, stride=4)
        self.upsampler_top = nn.ConvTranspose3d(embedding_dim, embedding_dim, 4, 2, 1)
        self.encoder_top = Encoder(channel, channel, num_res_block, num_res_channel, stride=2)
        self.encoder_conv_top = Conv(channel, embedding_dim, 1)
        self.quantizer_top = VectorQuantization3d(embedding_dim, num_embedding)
        self.decoder_top = Decoder(embedding_dim, embedding_dim, channel, num_res_block, num_res_channel, stride=2)

    def encode(self, input):
        encoded_bottom = self.encoder_bottom(input)
        encoded_top = self.encoder_top(encoded_bottom)
        encoded_top = self.encoder_conv_top(encoded_top)
        quantized_top, diff_top, idx_top = self.quantizer_top(encoded_top)
        decoded_top = self.decoder_top(quantized_top)
        encoded_bottom = torch.cat([decoded_top, encoded_bottom], dim=1)
        encoded_bottom = self.encoder_conv_bottom(encoded_bottom)
        quantized_bottom, diff_bottom, idx_bottom = self.quantizer_bottom(encoded_bottom)
        diff = (diff_top + diff_bottom) / 2
        return quantized_top, quantized_bottom, diff, idx_top, idx_bottom

    def decode(self, quantized_top, quantized_bottom):
        upsampled_top = self.upsampler_top(quantized_top)
        quantized = torch.cat([upsampled_top, quantized_bottom], dim=1)
        decoded = self.decoder_bottom(quantized)
        return decoded

    def decode_code(self, code_top, code_bottom):
        quantized_top = self.quantizer_top.embedding_code(code_top).permute(0, 4, 1, 2, 3)
        quantized_bottom = self.quantize_b.embedding_code(code_bottom).permute(0, 4, 1, 2, 3)
        decoded = self.decode(quantized_top, quantized_bottom)
        return decoded

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['uvw']
        quant_top, quant_bottom, vq_loss, output['idx_top'], output['idx_bottom'] = self.encode(x)
        decoded = self.decode(quant_top, quant_bottom)
        output['uvw'] = decoded
        output['loss'] = F.mse_loss(decoded, input['uvw']) + self.vq_commit * vq_loss
        return output


def vqvae():
    data_shape = cfg['data_shape']
    hidden_size = cfg['hidden_size']
    num_res_block = cfg['num_res_block']
    num_res_channel = cfg['num_res_channel']
    embedding_dim = cfg['embedding_dim']
    num_embedding = cfg['num_embedding']
    vq_commit = cfg['vq_commit']
    model = VQVAE(in_channel=data_shape[0], channel=hidden_size, num_res_block=num_res_block,
                  num_res_channel=num_res_channel, embedding_dim=embedding_dim, num_embedding=num_embedding,
                  vq_commit=vq_commit)
    model.apply(init_param)
    return model