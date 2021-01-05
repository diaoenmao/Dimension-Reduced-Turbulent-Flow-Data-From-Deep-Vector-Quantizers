import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from modules import VectorQuantization
from .utils import init_param, spectral_derivative_3d, weighted_mse_loss, physics


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
    def __init__(self, input_size, hidden_size, output_size, num_res_block, res_size, depth):
        super().__init__()
        if depth == 3:
            blocks = [
                nn.Conv3d(input_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size // 2, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size // 2, hidden_size, 4, 2, 1),
                nn.BatchNorm3d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size, hidden_size, 3, 1, 1),
            ]
        elif depth == 2:
            blocks = [
                nn.Conv3d(input_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size // 2, hidden_size, 4, 2, 1),
                nn.BatchNorm3d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size, hidden_size, 3, 1, 1),
            ]
        elif depth == 1:
            blocks = [
                nn.Conv3d(input_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size // 2, hidden_size, 3, 1, 1),
            ]
        else:
            raise ValueError('Not valid stride')
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size, res_size))
        blocks.extend([
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_size, output_size, 3, 1, 1)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_res_block, res_size, depth):
        super().__init__()
        blocks = [nn.Conv3d(input_size, hidden_size, 3, 1, 1)]
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size, res_size))
        if depth == 3:
            blocks.extend([
                nn.BatchNorm3d(hidden_size),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(hidden_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(hidden_size // 2, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(hidden_size // 2, output_size, 4, 2, 1),
            ])
        elif depth == 2:
            blocks.extend([
                nn.BatchNorm3d(hidden_size),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(hidden_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(hidden_size // 2, output_size, 4, 2, 1),
            ])
        elif depth == 1:
            blocks.extend([
                nn.BatchNorm3d(hidden_size),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(hidden_size, output_size, 4, 2, 1)
            ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(self, depth, hidden_size, embedding_size, num_embedding, num_res_block, res_size, vq_commit):
        super().__init__()
        self.encoder = Encoder(1, hidden_size, embedding_size, num_res_block, res_size, depth)
        self.quantizer = VectorQuantization(embedding_size, num_embedding, vq_commit)
        self.decoder = Decoder(embedding_size, hidden_size, 1, num_res_block, res_size, depth)

    def encode(self, input):
        x = input.view(input.size(0) * input.size(1), -1, *input.size()[2:])
        encoded = self.encoder(x)
        quantized, diff, code = self.quantizer(encoded)
        quantized = quantized.view(input.size(0), input.size(1), *quantized.size()[1:])
        code = code.view(input.size(0), input.size(1), *code.size()[1:])
        return quantized, diff, code

    def decode(self, quantized):
        x = quantized.view(quantized.size(0) * quantized.size(1), *quantized.size()[2:])
        decoded = self.decoder(x)
        decoded = decoded.view(quantized.size(0), quantized.size(1), *decoded.size()[2:])
        return decoded

    def decode_code(self, code):
        x = code.view(code.size(0) * code.size(1), *code.size()[2:])
        quantized = self.quantizer.embedding_code(x)
        quantized = quantized.view(code.size(0), code.size(1), *quantized.size()[1:])
        decoded = self.decode(quantized)
        return decoded

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['uvw']
        quantized, diff, code = self.encode(x)
        output['code'] = code
        decoded = self.decode(quantized)
        output['uvw'] = decoded
        output['loss'] = F.mse_loss(output['uvw'], input['uvw']) + diff
        for i in range(len(cfg['loss_mode'])):
            if 'duvw' not in output:
                if cfg['loss_commit'][i] > 0 or not self.training:
                    output['duvw'] = spectral_derivative_3d(output['uvw'])
            if cfg['loss_commit'][i] > 0:
                if cfg['loss_mode'][i] == 'exact':
                    output['loss'] += cfg['loss_commit'][i] * weighted_mse_loss(output['duvw'], input['duvw'])
                elif cfg['loss_mode'][i] == 'physics':
                    output['loss'] += cfg['loss_commit'][i] * physics(output['duvw'])
                else:
                    raise ValueError('Not valid loss mode')
        return output


def vqvae():
    depth = cfg['vqvae']['depth']
    hidden_size = cfg['vqvae']['hidden_size']
    embedding_size = cfg['vqvae']['embedding_size']
    num_embedding = cfg['vqvae']['num_embedding']
    num_res_block = cfg['vqvae']['num_res_block']
    res_size = cfg['vqvae']['res_size']
    vq_commit = cfg['vqvae']['vq_commit']
    model = VQVAE(depth, hidden_size, embedding_size, num_embedding, num_res_block, res_size, vq_commit)
    model.apply(init_param)
    return model
