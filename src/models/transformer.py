import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param
from config import cfg


class Conv3d(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv = nn.Conv3d(input_size, output_size, 3, 1, 1)

    def forward(self, input):
        N, S = input.size(0), input.size(1)
        x = self.conv(input.view(-1, *input.size()[2:]).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1).view(
            N, S, *input.size()[2:-1], -1)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.positional_embedding = nn.Embedding(sum(cfg['seq_length']) - 1, embedding_size)

    def forward(self, x):
        N, S, U, V, W = x.size()
        position = torch.arange(S, dtype=torch.long, device=x.device).view(1, -1, 1, 1, 1).expand((N, S, U, V, W))
        x = self.positional_embedding(position)
        return x


class TransformerEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_size):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.positional_embedding = PositionalEmbedding(embedding_size)
        self.embedding = nn.Embedding(num_embeddings, embedding_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_embedding(x)
        return x


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
        self.num_heads = num_heads
        self.map_q = Conv3d(embedding_size, embedding_size)
        self.map_k = Conv3d(embedding_size, embedding_size)
        self.map_v = Conv3d(embedding_size, embedding_size)
        self.map_o = Conv3d(embedding_size, embedding_size)
        self.attention = ScaledDotProduct(temperature=(embedding_size // num_heads) ** 0.5)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, U, V, W, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, U, V, W, self.num_heads, sub_dim).permute(0, 5, 2, 3, 4, 1, 6) \
            .reshape(batch_size * self.num_heads, U, V, W, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, U, V, W, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, U, V, W, seq_len, in_feature).permute(0, 5, 2, 3, 4, 1, 6) \
            .reshape(batch_size, seq_len, U, V, W, out_dim)

    def forward(self, q, k, v, mask=None):
        N, _, U, V, W, _ = q.size()
        q, k, v = self.map_q(q), self.map_k(k), self.map_v(v)
        q, k, v = self._reshape_to_batches(q), self._reshape_to_batches(k), self._reshape_to_batches(v)
        q, attn = self.attention(q, k, v, mask)
        q = self._reshape_from_batches(q)
        q = self.map_o(q)
        return q, attn


class Conv(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout):
        super().__init__()
        self.map_1 = Conv3d(embedding_size, hidden_size)
        self.map_2 = Conv3d(hidden_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.map_2(self.dropout(F.gelu(self.map_1(x))))
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout):
        super().__init__()
        self.norm_1 = nn.LayerNorm(embedding_size)
        self.norm_2 = nn.LayerNorm(embedding_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.mha = MultiheadAttention(embedding_size, num_heads)
        self.map = Conv(embedding_size, hidden_size, dropout)

    def forward(self, src, src_mask=None):
        _src, _ = self.mha(src, src, src, mask=src_mask)
        src = src + self.dropout_1(_src)
        src = self.norm_1(src)
        _src = self.map(src)
        src = src + self.dropout_2(_src)
        src = self.norm_2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, embedding, encoder_layer, num_layers):
        super().__init__()
        self.embedding = embedding
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, src_mask=None):
        x = self.embedding(src)
        for i in range(self.num_layers):
            x = self.layers[i](x, src_mask)
        return x


class Classifier(nn.Module):
    def __init__(self, embedding_size, num_embedding):
        super().__init__()
        self.linear_1 = nn.Linear(embedding_size, embedding_size)
        self.norm = nn.LayerNorm(embedding_size)
        self.linear_2 = nn.Linear(embedding_size, num_embedding)

    def forward(self, input):
        x = self.linear_2(F.gelu(self.norm(self.linear_1(input)))).permute(0, 6, 1, 2, 3, 4, 5)
        return x


class Transformer(nn.Module):
    def __init__(self, num_embedding, embedding_size, hidden_size, num_heads, num_layers, dropout):
        super().__init__()
        embedding = TransformerEmbedding(num_embedding, embedding_size)
        encoder_layer = TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(embedding, encoder_layer, num_layers)
        self.classifier = Classifier(embedding_size, num_embedding)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, 0).to(cfg['device'])
        mask[:, :cfg['seq_length'][0]] = 1
        return mask

    def next(self, input, n):
        x_ = input
        y = []
        for i in range(n):
            x = x_
            N, S, C, U, V, W = x.size()
            x = x.permute(0, 2, 1, 3, 4, 5)
            x = x.view(-1, S, U, V, W)
            self.src_mask = self._generate_square_subsequent_mask(S)
            x = self.transformer_encoder(x, self.src_mask)
            x = x[:, [-1]]
            x = x.view(N, C, 1, U, V, W, -1)
            x = x.permute(0, 2, 1, 3, 4, 5, 6)
            x = self.classifier(x)
            _x = x.topk(1, 1, True, True)[1][:, 0]
            y.append(_x)
            x_ = torch.cat([x_, _x], dim=1)
            x_ = x_[:, :(sum(cfg['seq_length']) - 1)]
        output = torch.cat(y, dim=1)
        return output

    def forward(self, input):
        output = {}
        x = torch.cat([input['code'], input['ncode'][:, :-1]], dim=1)
        N, S, C, U, V, W = x.size()
        x = x.permute(0, 2, 1, 3, 4, 5)
        x = x.view(-1, S, U, V, W)
        self.src_mask = self._generate_square_subsequent_mask(S)
        x = self.transformer_encoder(x, self.src_mask)
        x = x[:, -input['ncode'].size(1):]
        x = x.view(N, C, input['ncode'].size(1), U, V, W, -1)
        x = x.permute(0, 2, 1, 3, 4, 5, 6)
        x = self.classifier(x)
        output['score'] = x
        output['loss'] = F.cross_entropy(output['score'], input['ncode'])
        output['ncode'] = output['score'].topk(1, 1, True, True)[1][:, 0]
        return output


def transformer():
    num_embedding = cfg[cfg['ae_name']]['num_embedding']
    embedding_size = cfg['transformer']['embedding_size']
    hidden_size = cfg['transformer']['hidden_size']
    num_heads = cfg['transformer']['num_heads']
    num_layers = cfg['transformer']['num_layers']
    dropout = cfg['transformer']['dropout']
    model = Transformer(num_embedding, embedding_size, hidden_size, num_heads, num_layers, dropout)
    model.apply(init_param)
    return model
