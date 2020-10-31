import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param
from torch.nn import TransformerEncoder
from config import cfg


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.positional_embedding = nn.Embedding(2 * cfg['bptt'], embedding_size)

    def forward(self, x):
        N, S, H, W, D = x.size()
        position = torch.arange(S, dtype=torch.long, device=x.device).view(1, -1, 1, 1, 1).expand((N, S, H, W, D))
        x = self.positional_embedding(position)
        return x


class TransformerEmbedding(nn.Module):
    def __init__(self, num_tokens, embedding_size, dropout):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.positional_embedding = PositionalEmbedding(embedding_size)
        self.embedding = nn.Embedding(num_tokens, embedding_size)
        self.norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.embedding(src) + self.positional_embedding(src)
        src = self.dropout(self.norm(src))
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
        self.conv_q = nn.Conv3d(embedding_size, embedding_size, 3, 1, 1)
        self.conv_k = nn.Conv3d(embedding_size, embedding_size, 3, 1, 1)
        self.conv_v = nn.Conv3d(embedding_size, embedding_size, 3, 1, 1)
        self.conv_o = nn.Conv3d(embedding_size, embedding_size, 3, 1, 1)
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

    def _reshape_to_conv3d(self, x):
        return x.reshape(-1, *x.size()[2:]).permute(0, 4, 1, 2, 3)

    def _reshape_from_conv3d(self, x, N):
        return x.permute(0, 2, 3, 4, 1).reshape(N, -1, *x.size()[2:], x.size(1))

    def forward(self, q, k, v, mask=None):
        N, _, H, W, D, _ = q.size()
        q, k, v = self._reshape_to_conv3d(q), self._reshape_to_conv3d(k), self._reshape_to_conv3d(v)
        q, k, v = self.conv_q(q), self.conv_k(k), self.conv_v(v)
        q, k, v = self._reshape_from_conv3d(q, N), self._reshape_from_conv3d(k, N), self._reshape_from_conv3d(v, N)
        q, k, v = self._reshape_to_batches(q), self._reshape_to_batches(k), self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.num_heads, H, W, D, 1, 1)
        q, attn = self.attention(q, k, v, mask)
        q = self._reshape_from_batches(q)
        q = self._reshape_to_conv3d(q)
        q = self.conv_o(q)
        q = self._reshape_from_conv3d(q, N)
        return q, attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout):
        super().__init__()
        self.mha = MultiheadAttention(embedding_size, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, embedding_size)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.activation = nn.GELU()
        self.init_param()

    def init_param(self):
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.weight.data.fill_(1.0)
        self.norm1.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        return

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, _ = self.mha(src, src, src, mask=src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Decoder(nn.Module):
    def __init__(self, num_tokens, embedding_size):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, embedding_size)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(embedding_size)
        self.linear2 = nn.Linear(embedding_size, num_tokens)

    def forward(self, src):
        out = self.linear2(self.norm1(self.activation(self.linear1(src))))
        return out


class Transformer(nn.Module):
    def __init__(self, num_embedding, embedding_size, num_heads, hidden_size, num_layers, dropout):
        super().__init__()
        self.num_embedding = num_embedding
        self.transformer_embedding = TransformerEmbedding(num_embedding, embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.decoder = Decoder(num_embedding, embedding_size)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask[:, :sz // 2] = True
        mask = mask.float().masked_fill(mask == 0, float('-inf')).to(cfg['device'])
        return mask

    def forward(self, input):
        output = {}
        src = torch.cat([input['code'], input['ncode'][:,:-1]], dim=1)
        self.src_mask = self._generate_square_subsequent_mask(src.size(1))
        src = self.transformer_embedding(src)
        src = self.transformer_encoder(src, self.src_mask)
        out = self.decoder(src)
        output['score'] = out.permute(0, 5, 1, 2, 3, 4)[:, :, -cfg['bptt']:]
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