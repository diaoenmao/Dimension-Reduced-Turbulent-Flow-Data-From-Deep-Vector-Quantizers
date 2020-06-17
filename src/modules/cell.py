import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ntuple


def make_cell(cell_info):
    if cell_info['cell'] == 'none':
        cell = nn.Identity()
    elif cell_info['cell'] == 'Normalization':
        cell = Normalization(cell_info['mode'], cell_info['input_size'], cell_info['dim'])
    elif cell_info['cell'] == 'Activation':
        cell = Activation(cell_info['mode'])
    elif cell_info['cell'] == 'ResizeCell':
        cell = ResizeCell(cell_info)
    elif cell_info['cell'] == 'LinearCell':
        cell = LinearCell(cell_info)
    elif cell_info['cell'] == 'Conv3dCell':
        cell = Conv3dCell(cell_info)
    elif cell_info['cell'] == 'ConvTranspose3dCell':
        cell = ConvTranspose3dCell(cell_info)
    elif cell_info['cell'] == 'ResConv3dCell':
        cell = ResConv3dCell(cell_info)
    elif cell_info['cell'] == 'VectorQuantization3dCell':
        cell = VectorQuantization3dCell(cell_info)
    else:
        raise ValueError('Not valid cell info: {}'.format(cell_info))
    return cell


def Normalization(mode, size, dim=3):
    if mode == 'none':
        return nn.Identity()
    elif mode == 'bn':
        if dim == 1:
            return nn.BatchNorm1d(size)
        elif dim == 2:
            return nn.BatchNorm2d(size)
        elif dim == 3:
            return nn.BatchNorm3d(size)
    elif mode == 'in':
        return nn.InstanceNorm3d(size)
    elif mode == 'ln':
        return nn.LayerNorm(size)
    else:
        raise ValueError('Not valid normalization')
    return


def Activation(mode, inplace=True):
    if mode == 'none':
        return nn.Identity()
    elif mode == 'tanh':
        return nn.Tanh()
    elif mode == 'hardtanh':
        return nn.Hardtanh()
    elif mode == 'relu':
        return nn.ReLU(inplace=inplace)
    elif mode == 'prelu':
        return nn.PReLU()
    elif mode == 'elu':
        return nn.ELU(inplace=inplace)
    elif mode == 'selu':
        return nn.SELU(inplace=inplace)
    elif mode == 'celu':
        return nn.CELU(inplace=inplace)
    elif mode == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif mode == 'sigmoid':
        return nn.Sigmoid()
    elif mode == 'softmax':
        return nn.SoftMax()
    else:
        raise ValueError('Not valid activation')
    return


class ResizeCell(nn.Module):
    def __init__(self, cell_info):
        super(ResizeCell, self).__init__()
        default_cell_info = {}
        self.cell_info = {**default_cell_info, **cell_info}

    def forward(self, input):
        return input.view(input.size(0), *self.cell_info['resize'])


class LinearCell(nn.Linear):
    def __init__(self, cell_info):
        default_cell_info = {'bias': True}
        self.cell_info = {**default_cell_info, **cell_info}
        super(LinearCell, self).__init__(self.cell_info['input_size'], self.cell_info['output_size'],
                                         bias=self.cell_info['bias'])
        self.normalization = Normalization(self.cell_info['normalization'], self.cell_info['output_size'], 1)
        self.activation = Activation(self.cell_info['activation'])

    def forward(self, input):
        return self.activation(self.normalization(F.linear(input, self.weight, self.bias)))

    def forward(self, input):
        return self.mc(self.activation(self.normalization(F.linear(input, self.weight, self.bias))))


class Conv3dCell(nn.Conv3d):
    def __init__(self, cell_info):
        default_cell_info = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros'}
        self.cell_info = {**default_cell_info, **cell_info}
        super(Conv3dCell, self).__init__(self.cell_info['input_size'], self.cell_info['output_size'],
                                         self.cell_info['kernel_size'], stride=self.cell_info['stride'],
                                         padding=self.cell_info['padding'], dilation=self.cell_info['dilation'],
                                         groups=self.cell_info['groups'], bias=self.cell_info['bias'],
                                         padding_mode=self.cell_info['padding_mode'])
        self.normalization = Normalization(self.cell_info['normalization'], self.cell_info['output_size'])
        self.activation = Activation(self.cell_info['activation'])

    def forward(self, input):
        _tuple = ntuple(2)
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2,
                                (self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return self.activation(self.normalization(F.conv3d(F.pad(input, expanded_padding, mode='circular'),
                                                               self.weight, self.bias, self.stride, _tuple(0),
                                                               self.dilation, self.groups)))
        return self.activation(self.normalization(F.conv3d(input, self.weight, self.bias, self.stride,
                                                           self.padding, self.dilation, self.groups)))


class ConvTranspose3dCell(nn.ConvTranspose3d):
    def __init__(self, cell_info):
        default_cell_info = {'stride': 1, 'padding': 0, 'output_padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros'}
        self.cell_info = {**default_cell_info, **cell_info}
        super(ConvTranspose3dCell, self).__init__(self.cell_info['input_size'], self.cell_info['output_size'],
                                                  cell_info['kernel_size'], stride=self.cell_info['stride'],
                                                  padding=self.cell_info['padding'],
                                                  output_padding=self.cell_info['output_padding'],
                                                  dilation=self.cell_info['dilation'], groups=self.cell_info['groups'],
                                                  bias=self.cell_info['bias'],
                                                  padding_mode=self.cell_info['padding_mode'])
        self.normalization = Normalization(self.cell_info['normalization'], self.cell_info['output_size'])
        self.activation = Activation(self.cell_info['activation'])

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose3d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return self.activation(self.normalization(F.conv_transpose3d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)))


class ResConv3dCell(nn.Module):
    def __init__(self, cell_info):
        super(ResConv3dCell, self).__init__()
        default_cell_info = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros', 'mode': 'pass'}
        self.cell_info = {**default_cell_info, **cell_info}
        conv_1_info = {**self.cell_info, 'output_size': self.cell_info['hidden_size']}
        conv_2_info = {**self.cell_info, 'input_size': self.cell_info['hidden_size'], 'activation': 'none'}
        self.conv_1 = Conv3dCell(conv_1_info)
        self.conv_2 = Conv3dCell(conv_2_info)
        if self.cell_info['input_size'] != self.cell_info['output_size']:
            self.shortcut = Conv3dCell(
                {**self.cell_info, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'activation': 'none'})
        else:
            self.shortcut = nn.Identity()
        self.activation = Activation(self.cell_info['activation'])

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2, mode='trilinear', align_corners=False) \
            if self.cell_info['mode'] == 'up' else input
        shortcut = self.shortcut(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.activation(x + shortcut)
        output = F.avg_pool3d(x, 2) if self.cell_info['mode'] == 'down' else x
        return output


class VectorQuantization3dCell(nn.Module):
    def __init__(self, cell_info):
        default_cell_info = {'decay': 0.99, 'eps': 1e-5}
        self.cell_info = {**default_cell_info, **cell_info}
        super(VectorQuantization3dCell, self).__init__()
        self.embedding_dim = self.cell_info['embedding_dim']
        self.num_embedding = self.cell_info['num_embedding']
        self.decay = self.cell_info['decay']
        self.eps = self.cell_info['eps']
        embedding = torch.randn(self.embedding_dim, self.num_embedding)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(self.num_embedding))
        self.register_buffer('embedding_avg', embedding.clone())

    def forward(self, input):
        input = input.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = input.size()
        flatten = input.view(-1, self.embedding_dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embedding
                + self.embedding.pow(2).sum(0, keepdim=True)
        )
        _, embedding_ind = dist.min(1)
        embedding_onehot = F.one_hot(embedding_ind, self.num_embedding).type(flatten.dtype)
        embedding_ind = embedding_ind.view(*input_shape[:-1])
        quantize = self.embedding_code(embedding_ind)
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embedding_onehot.sum(0)
            )
            embedding_sum = flatten.transpose(0, 1) @ embedding_onehot
            self.embedding_avg.data.mul_(self.decay).add_(1 - self.decay, embedding_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.num_embedding * self.eps) * n
            )
            embedding_normalized = self.embedding_avg / cluster_size.unsqueeze(0)
            self.embedding.data.copy_(embedding_normalized)
        diff = F.mse_loss(quantize.detach(), input)
        quantize = input + (quantize - input).detach()
        quantize = quantize.permute(0, 4, 1, 2, 3).contiguous()
        return quantize, diff, embedding_ind

    def embedding_code(self, embedding_ind):
        return F.embedding(embedding_ind, self.embedding.transpose(0, 1))