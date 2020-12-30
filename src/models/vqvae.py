import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from modules import VectorQuantization
from .utils import init_param, spectral_derivative_3d, physics


class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_size, hidden_size, 3, 1, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_res_block, depth):
        super().__init__()
        blocks = [nn.Conv3d(input_size, hidden_size // (2 ** depth), 3, 1, 1)]
        for i in range(depth):
            blocks.extend([
                nn.BatchNorm3d(hidden_size // (2 ** (depth - i))),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size // (2 ** (depth - i)), hidden_size // (2 ** (depth - i - 1)), 4, 2, 1)
            ])
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size))
        blocks.extend([
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_size, output_size, 3, 1, 1)
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        out = self.blocks(input)
        return out


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_res_block, depth):
        super().__init__()
        blocks = [nn.Conv3d(input_size, hidden_size, 3, 1, 1)]
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size))
        for i in range(depth):
            blocks.extend([
                nn.BatchNorm3d(hidden_size // (2 ** i)),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(hidden_size // (2 ** i), hidden_size // (2 ** (i + 1)), 4, 2, 1)
            ])
        blocks.extend([
            nn.BatchNorm3d(hidden_size // (2 ** depth)),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_size // (2 ** depth), output_size, 3, 1, 1)
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        out = self.blocks(input)
        return out


class VQVAE(nn.Module):
    def __init__(self, depth, hidden_size, num_res_block, embedding_size, num_embedding, vq_commit):
        super().__init__()
        self.encoder = Encoder(1, hidden_size, embedding_size, num_res_block, depth)
        self.quantizer = VectorQuantization(embedding_size, num_embedding, vq_commit)
        self.decoder = Decoder(embedding_size, hidden_size, 1, num_res_block, depth)

    def encode(self, input):
        x = input
        encoded = self.encoder(x)
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
        N, C, U, V, W = x.size()
        x = x.view(N * C, 1, U, V, W)
        quantized, diff, code = self.encode(x)
        output['code'] = code.view(N, C, *code.size()[1:])
        decoded = self.decode(quantized)
        output['uvw'] = decoded.view(N, C, U, V, W)
        output['loss'] = F.mse_loss(output['uvw'], input['uvw']) + diff
        for i in range(len(cfg['loss_mode'])):
            if cfg['loss_commit'][i] > 0:
                if 'duvw' not in output:
                    output['duvw'] = spectral_derivative_3d(output['uvw'])
                if cfg['loss_mode'][i] == 'exact':
                    output['loss'] += cfg['loss_commit'][i] * F.mse_loss(output['duvw'], input['duvw'])
                elif cfg['loss_mode'][i] == 'physics':
                    output['loss'] += cfg['loss_commit'][i] * physics(output['duvw'], input['duvw'])
                else:
                    raise ValueError('Not valid loss mode')
        return output


def vqvae():
    depth = cfg['vqvae']['depth']
    hidden_size = cfg['vqvae']['hidden_size']
    num_res_block = cfg['vqvae']['num_res_block']
    embedding_size = cfg['vqvae']['embedding_size']
    num_embedding = cfg['vqvae']['num_embedding']
    vq_commit = cfg['vqvae']['vq_commit']
    model = VQVAE(depth, hidden_size, num_res_block, embedding_size, num_embedding, vq_commit)
    model.apply(init_param)
    return model
