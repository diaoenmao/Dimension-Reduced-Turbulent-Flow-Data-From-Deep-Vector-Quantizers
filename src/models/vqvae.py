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
    def __init__(self, input_size=3, hidden_size=128, num_res_block=2, res_size=32, embedding_size=64,
                 num_embedding=512, vq_commit=0.25):
        super().__init__()
        self.encoder_bottom = Encoder(input_size, hidden_size, num_res_block, res_size, stride=2)
        self.encoder_conv_bottom = Conv(hidden_size + embedding_size, embedding_size, 1, 1, 0)
        self.quantizer_bottom = VectorQuantization3d(embedding_size, num_embedding)
        self.decoder_bottom = Decoder(embedding_size * 3, input_size, hidden_size, num_res_block, res_size, stride=2)

        self.upsampler_mid = nn.ConvTranspose3d(embedding_size, embedding_size, 4, 2, 1)
        self.encoder_mid = Encoder(hidden_size, hidden_size, num_res_block, res_size, stride=2)
        self.encoder_conv_mid = Conv(hidden_size + embedding_size, embedding_size, 1, 1, 0)
        self.quantizer_mid = VectorQuantization3d(embedding_size, num_embedding)
        self.decoder_mid = Decoder(embedding_size, embedding_size, hidden_size, num_res_block, res_size, stride=2)

        self.upsampler_top = nn.ConvTranspose3d(embedding_size, embedding_size, 4, 2, 1)
        self.encoder_top = Encoder(hidden_size, hidden_size, num_res_block, res_size, stride=2)
        self.encoder_conv_top = Conv(hidden_size, embedding_size, 1, 1, 0)
        self.quantizer_top = VectorQuantization3d(embedding_size, num_embedding)
        self.decoder_top = Decoder(embedding_size, embedding_size, hidden_size, num_res_block, res_size, stride=2)
        self.vq_commit = vq_commit

    def encode(self, input):
        encoded_bottom = self.encoder_bottom(input)
        encoded_mid = self.encoder_mid(encoded_bottom)
        encoded_top = self.encoder_top(encoded_mid)

        encoded_top = self.encoder_conv_top(encoded_top)
        quantized_top, diff_top, idx_top = self.quantizer_top(encoded_top)
        decoded_top = self.decoder_top(quantized_top)

        encoded_mid = torch.cat([decoded_top, encoded_mid], dim=1)
        encoded_mid = self.encoder_conv_mid(encoded_mid)
        quantized_mid, diff_mid, idx_mid = self.quantizer_mid(encoded_mid)
        decoded_mid = self.decoder_mid(quantized_mid)

        encoded_bottom = torch.cat([decoded_mid, encoded_bottom], dim=1)
        encoded_bottom = self.encoder_conv_bottom(encoded_bottom)
        quantized_bottom, diff_bottom, idx_bottom = self.quantizer_bottom(encoded_bottom)
        return (quantized_top, quantized_mid, quantized_bottom), \
               (diff_top, diff_mid, diff_bottom), (idx_top, idx_mid, idx_bottom)

    def decode(self, quantized):
        quantized_top, quantized_mid, quantized_bottom = quantized
        upsampled_top = self.upsampler_top(quantized_top)
        upsampled_top = self.upsampler_mid(upsampled_top)
        upsampled_mid = self.upsampler_mid(quantized_mid)
        quantized = torch.cat([upsampled_top, upsampled_mid, quantized_bottom], dim=1)
        decoded = self.decoder_bottom(quantized)
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
    num_res_block = cfg['num_res_block']
    res_size = cfg['res_size']
    embedding_size = cfg['embedding_size']
    num_embedding = cfg['num_embedding']
    vq_commit = cfg['vq_commit']
    model = VQVAE(input_size=data_shape[0], hidden_size=hidden_size, num_res_block=num_res_block,
                  res_size=res_size, embedding_size=embedding_size, num_embedding=num_embedding,
                  vq_commit=vq_commit)
    model.apply(init_param)
    return model