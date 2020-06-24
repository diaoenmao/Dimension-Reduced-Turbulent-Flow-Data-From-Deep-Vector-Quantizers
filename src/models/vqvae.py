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


class DownBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.down = nn.Sequential(
            Normalization(hidden_size),
            Activation(inplace=True),
            Conv(hidden_size, hidden_size, 4, 2, 1),
        )

    def forward(self, input):
        out = self.down(input)
        return out


class UpBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.up = nn.Sequential(
            Normalization(hidden_size),
            Activation(inplace=True),
            nn.ConvTranspose3d(hidden_size, hidden_size, 4, 2, 1),
        )

    def forward(self, input):
        out = self.up(input)
        return out


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, num_res_block, res_size, embedding_size, num_embedding):
        super().__init__()
        self.conv_in = Conv(input_size, hidden_size, 3, 1, 1)
        self.num_layer = num_layer
        self.down = nn.ModuleList([])
        self.res_blocks_down = nn.ModuleList([])
        self.conv_down = nn.ModuleList([])
        self.quantizer = nn.ModuleList([])
        self.up = nn.ModuleList([])
        self.res_blocks_up = nn.ModuleList([])
        self.conv_up = nn.ModuleList([])
        for i in range(num_layer):
            for _ in range(num_res_block):
                self.res_blocks_down.append(ResBlock(hidden_size, res_size))
            if i < num_layer - 1:
                self.down.append(DownBlock(hidden_size))
                self.conv_down.append(nn.Sequential(Normalization(hidden_size),
                                                    Activation(inplace=True),
                                                    Conv(hidden_size + embedding_size, embedding_size, 3, 1, 1)))
            else:
                self.conv_down.append(nn.Sequential(Normalization(hidden_size),
                                                    Activation(inplace=True),
                                                    Conv(hidden_size, embedding_size, 3, 1, 1)))
            if i > 0:
                self.conv_up.append(Conv(embedding_size, hidden_size, 3, 1, 1))
                for _ in range(num_res_block):
                    self.res_blocks_up.append(ResBlock(hidden_size, res_size))
                self.up.append(UpBlock(hidden_size))
            self.quantizer.append(VectorQuantization3d(embedding_size, num_embedding))

    def decode(self, code):
        encoded = []
        for i in range(self.num_layer):
            encoded.append(self.quantizer[i].embedding_code(code[i]).permute(0, 4, 1, 2, 3))
        return encoded

    def forward(self, input):
        x = self.conv_in(input)
        res_down = [None for _ in range(self.num_layer)]
        encoded = [None for _ in range(self.num_layer)]
        diff = [None for _ in range(self.num_layer)]
        code = [None for _ in range(self.num_layer)]
        for i in range(self.num_layer):
            res_down[i] = self.res_blocks_down[i](x)
            if i < self.num_layer - 1:
                x = self.down[i](res_down[i])
        for i in range(self.num_layer, 0, -1):
            if i < self.num_layer - 1:
                x = self.conv_merge[i](torch.cat([res_down[i], res_up], dim=1))
            else:
                x = res_down[i]
            encoded[i], diff[i], code[i] = self.quantizer[i](x)
            if i > 0:
                res_up = self.up[i](self.res_blocks_up[i](self.conv_up[i](encoded[i])))
        return encoded, diff, code


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_res_block, res_size):
        super().__init__()
        self.conv_merge = Conv(input_size, hidden_size, 3, 1, 1)
        self.res_blocks = nn.ModuleList([])
        for _ in range(num_res_block):
            self.res_blocks.append(ResBlock(hidden_size, res_size))
        self.conv_out = Conv(hidden_size, output_size, 3, 1, 1)

    def forward(self, input):
        x = self.conv_merge(torch.cat(input, dim=1))
        x = self.res_blocks(x)
        decoded = self.conv_out(x)
        return decoded


class VQVAE(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, depth=3, num_res_block=2, res_size=32, embedding_size=64,
                 num_embedding=512, vq_commit=0.25):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_res_block = num_res_block
        self.res_size = res_size
        self.embedding_size = embedding_size
        self.num_embedding = num_embedding
        self.vq_commit = vq_commit
        self.encoder = Encoder(input_size, hidden_size, depth, num_res_block, res_size, embedding_size, num_embedding)
        self.decoder = Decoder(embedding_size * depth, hidden_size, input_size, num_res_block, res_size)

    def encode(self, input):
        encoded, diff, code = self.encoder(input)
        return encoded, diff, code

    def decode(self, encoded):
        decoded = self.decoder(encoded)
        return decoded

    def decode(self, code):
        encoded = self.encoder.decode(code)
        decoded = self.decode(encoded)
        return decoded

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['uvw']
        encoded, diff, output['code'] = self.encode(x)
        vq_loss = sum(diff) / len(diff)
        decoded = self.decode(encoded)
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
                  res_size=res_size, embedding_size=embedding_size, num_embedding=num_embedding, vq_commit=vq_commit)
    model.apply(init_param)
    return model