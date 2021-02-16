import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param
from config import cfg

Normalization = nn.LayerNorm #BatchNorm3d
Activation = nn.GELU


def _reshape_to_conv3d(x):
    return x.reshape(-1, *x.size()[2:]).permute(0, 4, 1, 2, 3)


def _reshape_from_conv3d(x, N):
    return x.permute(0, 2, 3, 4, 1).reshape(N, -1, *x.size()[2:], x.size(1))


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.positional_embedding = nn.Embedding(cfg['bptt'] + cfg['pred_length'], embedding_size)

    def forward(self, x):
        N, S, H, W, D = x.size()
        position = torch.arange(S, dtype=torch.long, device=x.device).view(1, -1, 1, 1, 1).expand((N, S, H, W, D))
        x = self.positional_embedding(position)
        return x


class TransformerEmbedding(nn.Module):
    def __init__(self, num_tokens, embedding_size):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.positional_embedding = PositionalEmbedding(embedding_size)
        self.embedding = nn.Embedding(num_tokens, embedding_size)

    def forward(self, src):
        src = self.embedding(src) + self.positional_embedding(src)
        return src


class ScaledDotProduct(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        scores = q.matmul(k.transpose(-2, -1)) / self.temperature
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.map_q = nn.Conv3d(embedding_size, embedding_size, 3, 1, 1)
        self.map_k = nn.Conv3d(embedding_size, embedding_size, 3, 1, 1)
        self.map_v = nn.Conv3d(embedding_size, embedding_size, 3, 1, 1)
        self.map_o = nn.Conv3d(embedding_size, embedding_size, 3, 1, 1)
        self.attention = ScaledDotProduct(temperature=(embedding_size // num_heads) ** 0.5)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, H, W, D, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, H, W, D, self.num_heads, sub_dim).permute(0, 5, 2, 3, 4, 1, 6) \
            .reshape(batch_size * self.num_heads, H, W, D, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, H, W, D, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, H, W, D, seq_len, in_feature).permute(0, 5, 2, 3, 4, 1, 6) \
            .reshape(batch_size, seq_len, H, W, D, out_dim)

    def forward(self, q, k, v, mask=None):
        N, _, H, W, D, _ = q.size()
        q, k, v = _reshape_to_conv3d(q), _reshape_to_conv3d(k), _reshape_to_conv3d(v)
        q, k, v = self.map_q(q), self.map_k(k), self.map_v(v)
        q, k, v = _reshape_from_conv3d(q, N), _reshape_from_conv3d(k, N), _reshape_from_conv3d(v, N)
        q, k, v = self._reshape_to_batches(q), self._reshape_to_batches(k), self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.num_heads, H, W, D, 1, 1)
        q, attn = self.attention(q, k, v, mask)
        q = _reshape_to_conv3d(self._reshape_from_batches(q))
        q = _reshape_from_conv3d(self.map_o(q), N)
        return q, attn


class Conv(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout):
        super().__init__()
        self.map_1 = nn.Conv3d(embedding_size, hidden_size, 3, 1, 1)
        self.map_2 = nn.Conv3d(hidden_size, embedding_size, 3, 1, 1)
        self.activation = Activation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N = x.size()[0]
        x = self.dropout(self.activation(_reshape_from_conv3d((self.map_1(_reshape_to_conv3d(x))), N)))        
        x = _reshape_from_conv3d(self.map_2(_reshape_to_conv3d(x)), N)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout):
        super().__init__()
        self.norm_1 = Normalization(embedding_size)
        self.norm_2 = Normalization(embedding_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.mha = MultiheadAttention(embedding_size, num_heads)
        self.conv = Conv(embedding_size, hidden_size, dropout)

    def forward(self, src, src_mask=None):
        N = src.size()[0]
        _src = self.norm_1((src))
        _src, _ = self.mha(_src, _src, _src, mask=src_mask)
        src = src + self.dropout_1(_src)
        _src = self.norm_2((src))
        _src = self.conv(_src)
        src = src + self.dropout_2(_src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, embedding, encoder_layer, num_layers, embedding_size):
        super().__init__()
        self.embedding = embedding
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = Normalization(embedding_size)

    def forward(self, src, src_mask=None):
        x = self.embedding(src)
        for i in range(self.num_layers):
            x = self.layers[i](x, src_mask)
        x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_tokens, embedding_size):
        super().__init__()
        self.conv = nn.Conv3d(embedding_size, embedding_size, 3, 1, 1)
        self.activation = Activation()
        self.norm = Normalization(embedding_size)
        self.linear = nn.Linear(embedding_size, num_tokens)

    def forward(self, src):
        N = src.size()[0]
        out = self.linear(self.norm(self.activation(_reshape_from_conv3d(self.conv(_reshape_to_conv3d(src)), N))))
        return out


class Transformer(nn.Module):
    def __init__(self, num_embedding, embedding_size, num_heads, hidden_size, num_layers, dropout):
        super().__init__()
        self.num_embedding = num_embedding
        embedding = TransformerEmbedding(num_embedding + 1, embedding_size)
        encoder_layer = TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(embedding, encoder_layer, num_layers, embedding_size)
        self.transformer_decoder = TransformerDecoder(num_embedding, embedding_size)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).to(cfg['device'])
        return mask

    def forward(self, input):
        output = {}
        src = input['code']
        mask = torch.tensor([self.num_embedding], dtype=torch.long,
                            device=src.device).expand(src.size(0), cfg['pred_length'], *src.size()[-3:])
        src = torch.cat([src, mask], dim=1)
        src = self.transformer_encoder(src, self.src_mask)
        out = self.transformer_decoder(src)
        output['score'] = out.permute(0, 5, 1, 2, 3, 4)[:, :, -cfg['pred_length']:]
        output['loss'] = F.cross_entropy(output['score'], input['ncode'])
        output['code'] = output['score'].topk(1, 1, True, True)[1][:, 0]
        return output


def transformer():
    num_embedding = cfg[cfg['ae_name']]['num_embedding']
    embedding_size = cfg['transformer']['embedding_size']
    num_heads = cfg['transformer']['num_heads']
    hidden_size = cfg['transformer']['hidden_size']
    num_layers = cfg['transformer']['num_layers']
    dropout = cfg['transformer']['dropout']
    model = Transformer(num_embedding, embedding_size, num_heads, hidden_size, num_layers, dropout)
    model.apply(init_param)
    return model