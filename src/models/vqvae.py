import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from modules import VectorQuantization
from .utils import init_param, spectral_derivative_3d, physics

Normalization = nn.BatchNorm3d
Activation = nn.ReLU


class ResBlock(nn.Module):
    def __init__(self, hidden_size, res_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_size, res_size, 3, 1, 1),
            nn.BatchNorm3d(res_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(res_size, hidden_size, 3, 1, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_res_block, res_size, stride):
        super().__init__()
        if stride == 4:
            blocks = [
                nn.Conv3d(input_size, hidden_size // 2, 4, 2, 1),
                Normalization(hidden_size // 2),
                Activation(inplace=True),
                nn.Conv3d(hidden_size // 2, hidden_size, 4, 2, 1),
                Normalization(hidden_size),
                Activation(inplace=True),
                nn.Conv3d(hidden_size, hidden_size, 3, 1, 1),
            ]
        elif stride == 2:
            blocks = [
                nn.Conv3d(input_size, hidden_size // 2, 4, 2, 1),
                Normalization(hidden_size // 2),
                Activation(inplace=True),
                nn.Conv3d(hidden_size // 2, hidden_size, 3, 1, 1),
            ]
        else:
            raise ValueError('Not valid stride')
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size, res_size))
        blocks.extend([
            Normalization(hidden_size),
            Activation(inplace=True)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_res_block, res_size, stride):
        super().__init__()
        blocks = [nn.Conv3d(input_size, hidden_size, 3, 1, 1)]
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size, res_size))
        if stride == 4:
            blocks.extend([
                Normalization(hidden_size),
                Activation(inplace=True),
                nn.ConvTranspose3d(hidden_size, hidden_size // 2, 4, 2, 1),
                Normalization(hidden_size // 2),
                Activation(inplace=True),
                nn.ConvTranspose3d(hidden_size // 2, output_size, 4, 2, 1),
            ])
        elif stride == 2:
            blocks.extend([
                Normalization(hidden_size),
                Activation(inplace=True),
                nn.ConvTranspose3d(hidden_size, output_size, 4, 2, 1)
            ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, depth=2, num_res_block=2, res_size=32, embedding_size=64,
                 num_embedding=512, d_mode='exact', d_commit=None, vq_commit=0.25):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_res_block, res_size, stride=2 ** depth)
        self.encoder_conv = nn.Conv3d(hidden_size, embedding_size, 1, 1, 0)
        self.quantizer = VectorQuantization(embedding_size, num_embedding, vq_commit)
        self.decoder = Decoder(embedding_size, input_size, hidden_size, num_res_block, res_size, stride=2 ** depth)
        self.d_mode = d_mode
        self.d_commit = d_commit

    def encode(self, input):
        x = input
        encoded = self.encoder(x)
        encoded = self.encoder_conv(encoded)
        quantized, diff, code = self.quantizer(encoded)
        return quantized, diff, code

    def decode(self, quantized):
        decoded = self.decoder(quantized)
        return decoded

    def decode_code(self, code):
        quantized = self.quantizer.embedding_code(code).transpose(1, -1).contiguous()
        decoded = self.decode(quantized)
        return decoded

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['uvw']
        quantized, diff, output['code'] = self.encode(x)
        decoded = self.decode(quantized)
        output['uvw'] = decoded
        output['duvw'] = spectral_derivative_3d(output['uvw'])
        output['loss'] = F.mse_loss(output['uvw'], input['uvw']) + diff
        for i in range(len(self.d_mode)):
            if self.d_mode[i] == 'exact':
                output['loss'] += self.d_commit[i] * F.mse_loss(output['duvw'], input['duvw'])
            elif self.d_mode[i] == 'physics':
                output['loss'] += self.d_commit[i] * physics(output['duvw'])
            else:
                raise ValueError('Not valid d_mode')
        return output


def vqvae():
    data_shape = cfg['data_shape']
    hidden_size = cfg['vqvae']['hidden_size']
    depth = cfg['vqvae']['depth']
    num_res_block = cfg['vqvae']['num_res_block']
    res_size = cfg['vqvae']['res_size']
    embedding_size = cfg['vqvae']['embedding_size']
    num_embedding = cfg['vqvae']['num_embedding']
    d_mode = cfg['d_mode']
    d_commit = cfg['d_commit']
    vq_commit = cfg['vqvae']['vq_commit']
    model = VQVAE(input_size=data_shape[0], hidden_size=hidden_size, depth=depth, num_res_block=num_res_block,
                  res_size=res_size, embedding_size=embedding_size, num_embedding=num_embedding,
                  d_mode=d_mode, d_commit=d_commit, vq_commit=vq_commit)
    model.apply(init_param)
    return model