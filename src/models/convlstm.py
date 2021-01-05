import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param
from config import cfg


class Embedding(nn.Module):
    def __init__(self, num_embedding, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embedding, hidden_size)

    def forward(self, input):
        x = self.embedding(input)
        return x


class Conv3d(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.conv = nn.Conv3d(hidden_size, 4 * hidden_size, 3, 1, 1)
        self.norm = nn.LayerNorm(4 * hidden_size)

    def forward(self, input):
        x = self.norm(self.conv(input.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1))
        return x


class Classifier(nn.Module):
    def __init__(self, hidden_size, num_embedding):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.linear_2 = nn.Linear(hidden_size, num_embedding)

    def forward(self, input):
        x = self.linear_2(F.gelu(self.norm(self.linear_1(input)))).permute(0, 6, 1, 2, 3, 4, 5)
        return x


class ConvLSTMBlock(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden = None
        self.blocks = nn.ModuleList([nn.ModuleDict({}) for _ in range(num_layers)])
        for i in range(num_layers):
            self.blocks[i]['in'] = Conv3d(hidden_size)
            self.blocks[i]['hidden'] = Conv3d(hidden_size)

    def init_hidden(self, hidden_size, dtype=torch.float):
        hidden = [torch.zeros(hidden_size, device=cfg['device'], dtype=dtype),
                  torch.zeros(hidden_size, device=cfg['device'], dtype=dtype)]
        return hidden

    def free_hidden(self):
        self.hidden = None
        return

    def forward(self, input, hidden=None):
        x = input
        if self.hidden is None:
            self.hidden = hidden
        hx, cx = None, None
        for i in range(self.num_layers):
            y = [None for _ in range(x.size(1))]
            for j in range(x.size(1)):
                gates = self.blocks[i]['in'](x[:, j])
                if hidden is None:
                    if self.hidden is None:
                        self.hidden = self.init_hidden((gates.size(0), *gates.size()[1:4], self.hidden_size))
                if j == 0:
                    hx, cx = self.hidden[0], self.hidden[1]
                gates += self.blocks[i]['hidden'](hx)
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=-1)
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = torch.tanh(cellgate)
                outgate = torch.sigmoid(outgate)
                cx = (forgetgate * cx) + (ingate * cellgate)
                hx = outgate * torch.tanh(cx)
                y[j] = hx
            x = torch.stack(y, dim=1)
        return x


class ConvLSTM(nn.Module):
    def __init__(self, num_embedding, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_embedding = num_embedding
        self.embedding = Embedding(num_embedding, hidden_size)
        self.block = ConvLSTMBlock(hidden_size, num_layers)
        self.classifier = Classifier(hidden_size, num_embedding)

    def next(self, input, n):
        x = input
        y = []
        for i in range(n):
            N, S, C, U, V, W = x.size()
            x = x.permute(0, 2, 1, 3, 4, 5)
            x = x.view(-1, S, U, V, W)
            x = self.embedding(x)
            x = self.block(x)
            x = x[:, [-1]]
            x = x.view(N, C, 1, U, V, W, -1)
            x = x.permute(0, 2, 1, 3, 4, 5, 6)
            x = self.classifier(x)
            x = x.topk(1, 1, True, True)[1][:, 0]
            y.append(x)
        self.block.free_hidden()
        output = torch.cat(y, dim=1)
        return output

    def forward(self, input):
        output = {}
        x = torch.cat([input['code'], input['ncode'][:, :-1]], dim=1)
        N, S, C, U, V, W = x.size()
        x = x.permute(0, 2, 1, 3, 4, 5)
        x = x.view(-1, S, U, V, W)
        x = self.embedding(x)
        x = self.block(x)
        self.block.free_hidden()
        x = x[:, -input['ncode'].size(1):]
        x = x.view(N, C, input['ncode'].size(1), U, V, W, -1)
        x = x.permute(0, 2, 1, 3, 4, 5, 6)
        x = self.classifier(x)
        output['score'] = x
        output['loss'] = F.cross_entropy(output['score'], input['ncode'])
        output['ncode'] = output['score'].topk(1, 1, True, True)[1][:, 0]
        return output


def convlstm():
    num_embedding = cfg[cfg['ae_name']]['num_embedding']
    hidden_size = cfg['convlstm']['hidden_size']
    num_layers = cfg['convlstm']['num_layers']
    model = ConvLSTM(num_embedding, hidden_size, num_layers)
    model.apply(init_param)
    return model
