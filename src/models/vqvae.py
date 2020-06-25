import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from modules import VectorQuantization
from .utils import init_param

Normalization = nn.BatchNorm3d
Activation = nn.ReLU
Conv = nn.Conv3d


class ResBlock(nn.Module):
    def __init__(self, hidden_size, res_size):
        super().__init__()
        self.conv = nn.Sequential(
            Normalization(hidden_size),
            Activation(inplace=True),
            Conv(hidden_size, res_size, 3, 1, 1),
            Normalization(res_size),
            Activation(inplace=True),
            Conv(res_size, hidden_size, 3, 1, 1),
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
                Conv(input_size, hidden_size // 2, 4, 2, 1),
                Normalization(hidden_size // 2),
                Activation(inplace=True),
                Conv(hidden_size // 2, hidden_size, 4, 2, 1),
                Normalization(hidden_size),
                Activation(inplace=True),
                Conv(hidden_size, hidden_size, 3, 1, 1),
            ]
        elif stride == 2:
            blocks = [
                Conv(input_size, hidden_size // 2, 4, 2, 1),
                Normalization(hidden_size // 2),
                Activation(inplace=True),
                Conv(hidden_size // 2, hidden_size, 3, 1, 1),
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
        blocks = [Conv(input_size, hidden_size, 3, 1, 1)]
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size, res_size))
        blocks.extend([
            Normalization(hidden_size),
            Activation(inplace=True)])
        if stride == 4:
            blocks.extend([
                nn.ConvTranspose3d(hidden_size, hidden_size // 2, 4, 2, 1),
                Normalization(hidden_size // 2),
                Activation(inplace=True),
                nn.ConvTranspose3d(hidden_size // 2, output_size, 4, 2, 1),
            ])
        elif stride == 2:
            blocks.append(
                nn.ConvTranspose3d(hidden_size, output_size, 4, 2, 1)
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, depth=3, num_res_block=2, res_size=32, embedding_size=64,
                 num_embedding=512, vq_commit=0.25):
        super().__init__()
        self.upsampler, self.encoder, self.encoder_conv, self.quantizer, self.decoder = [], [], [], [], []
        for i in range(depth):
            if i == 0:
                self.upsampler.append(None)
                self.encoder.append(Encoder(input_size, hidden_size, num_res_block, res_size, stride=2))
                if depth > 1:
                    self.encoder_conv.append(Conv(hidden_size + embedding_size, embedding_size, 1, 1, 0))
                else:
                    self.encoder_conv.append(Conv(hidden_size, embedding_size, 1, 1, 0))
                self.quantizer.append(VectorQuantization(embedding_size, num_embedding))
                self.decoder.append(Decoder(embedding_size * depth, input_size, hidden_size,
                                            num_res_block, res_size, stride=2))
            else:
                self.upsampler.append(nn.ConvTranspose3d(embedding_size, embedding_size, 4, 2, 1))
                self.encoder.append(Encoder(hidden_size, hidden_size, num_res_block, res_size, stride=2))
                if i == depth - 1:
                    self.encoder_conv.append(Conv(hidden_size, embedding_size, 1, 1, 0))
                else:
                    self.encoder_conv.append(Conv(hidden_size + embedding_size, embedding_size, 1, 1, 0))
                self.quantizer.append(VectorQuantization(embedding_size, num_embedding))
                self.decoder.append(Decoder(embedding_size, embedding_size, hidden_size,
                                            num_res_block, res_size, stride=2))
        self.upsampler = nn.ModuleList(self.upsampler)
        self.encoder = nn.ModuleList(self.encoder)
        self.encoder_conv = nn.ModuleList(self.encoder_conv)
        self.quantizer = nn.ModuleList(self.quantizer)
        self.decoder = nn.ModuleList(self.decoder)
        self.depth = depth
        self.vq_commit = vq_commit

    def encode(self, input):
        encoded = [None for _ in range(self.depth)]
        quantized = [None for _ in range(self.depth)]
        diff = [None for _ in range(self.depth)]
        code = [None for _ in range(self.depth)]
        decoded = [None for _ in range(self.depth)]
        x = input
        for i in range(self.depth):
            encoded[i] = self.encoder[i](x)
            x = encoded[i]
        for i in range(self.depth - 1, -1, -1):
            if i < self.depth - 1:
                encoded[i] = torch.cat([decoded[i + 1], encoded[i]], dim=1)
            encoded[i] = self.encoder_conv[i](encoded[i])
            quantized[i], diff[i], code[i] = self.quantizer[i](encoded[i])
            if i > 0:
                decoded[i] = self.decoder[i](quantized[i])
        return quantized, diff, code

    def decode(self, quantized):
        upsampled = [None for _ in range(self.depth)]
        for i in range(self.depth - 1, -1, -1):
            upsampled[i] = quantized[i]
            for j in range(i, 0, -1):
                upsampled[i] = self.upsampler[j](upsampled[i])
        upsampled = torch.cat(upsampled, dim=1)
        decoded = self.decoder[0](upsampled)
        return decoded

    def decode_code(self, code):
        code_top, code_mid, code_bottom = code
        quantized_top = self.quantizer_top.embedding_code(code_top).permute(0, 4, 1, 2, 3)
        quantized_mid = self.quantizer_top.embedding_code(code_mid).permute(0, 4, 1, 2, 3)
        quantized_bottom = self.quantizer_bottom.embedding_code(code_bottom).permute(0, 4, 1, 2, 3)
        quantized = (quantized_top, quantized_mid, quantized_bottom)
        decoded = self.decode(quantized)
        return decoded

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['uvw']
        quantized, diff, output['code'] = self.encode(x)
        vq_loss = sum(diff) / len(diff)
        decoded = self.decode(quantized)
        output['uvw'] = decoded
        output['loss'] = F.mse_loss(decoded, input['uvw']) + self.vq_commit * vq_loss
        return output


def vqvae():
    data_shape = cfg['data_shape']
    hidden_size = cfg['hidden_size']
    depth = cfg['depth']
    num_res_block = cfg['num_res_block']
    res_size = cfg['res_size']
    embedding_size = cfg['embedding_size']
    num_embedding = cfg['num_embedding']
    vq_commit = cfg['vq_commit']
    model = VQVAE(input_size=data_shape[0], hidden_size=hidden_size, depth=depth, num_res_block=num_res_block,
                  res_size=res_size, embedding_size=embedding_size, num_embedding=num_embedding,
                  vq_commit=vq_commit)
    model.apply(init_param)
    return model