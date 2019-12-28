import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ntuple
from .shuffle import UnShuffle3d, Shuffle3d


def make_cell(cell_info):
    if cell_info['cell'] == 'none':
        cell = nn.Identity()
    elif cell_info['cell'] == 'Normalization':
        cell = Normalization(cell_info)
    elif cell_info['cell'] == 'Activation':
        cell = Activation(cell_info)
    elif cell_info['cell'] == 'LinearCell':
        cell = LinearCell(cell_info)
    elif cell_info['cell'] == 'Conv3dCell':
        cell = Conv3dCell(cell_info)
    elif cell_info['cell'] == 'Conv3dTransposeCell':
        cell = ConvTranspose3dCell(cell_info)
    elif cell_info['cell'] == 'ResConv3dCell':
        cell = ResConv3dCell(cell_info)
    elif cell_info['cell'] == 'Shuffle3dCell':
        cell = Shuffle3dCell(cell_info)
    else:
        raise ValueError('Not valid cell info')
    return cell


def Normalization(mode, size):
    if mode == 'none':
        return nn.Identity()
    elif mode == 'bn':
        return nn.BatchNorm2d(size)
    elif mode == 'in':
        return nn.InstanceNorm2d(size)
    elif mode == 'ln':
        return nn.LayerNorm(size)
    else:
        raise ValueError('Not valid normalization')
    return


def Activation(mode):
    if mode == 'none':
        return nn.Sequential()
    elif mode == 'tanh':
        return nn.Tanh()
    elif mode == 'hardtanh':
        return nn.Hardtanh()
    elif mode == 'relu':
        return nn.ReLU(inplace=True)
    elif mode == 'prelu':
        return nn.PReLU()
    elif mode == 'elu':
        return nn.ELU(inplace=True)
    elif mode == 'selu':
        return nn.SELU(inplace=True)
    elif mode == 'celu':
        return nn.CELU(inplace=True)
    elif mode == 'sigmoid':
        return nn.Sigmoid()
    elif mode == 'softmax':
        return nn.SoftMax()
    else:
        raise ValueError('Not valid activation')
    return


class LinearCell(nn.Linear):
    def __init__(self, cell_info):
        default_cell_info = {'bias': True}
        cell_info = {**default_cell_info, **cell_info}
        super(LinearCell, self).__init__(cell_info['input_size'], cell_info['output_size'], bias=cell_info['bias'])
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.normalization = Normalization(cell_info['normalization'], self.output_size)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input):
        return self.activation(self.normalization(F.linear(input, self.weight, self.bias)))

    def extra_repr(self):
        return str(self.cell_info)


class Conv3dCell(nn.Conv3d):
    def __init__(self, cell_info):
        default_cell_info = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros'}
        cell_info = {**default_cell_info, **cell_info}
        super(Conv3dCell, self).__init__(cell_info['input_size'], cell_info['output_size'], cell_info['kernel_size'],
                                         stride=cell_info['stride'], padding=cell_info['padding'],
                                         dilation=cell_info['dilation'], groups=cell_info['groups'],
                                         bias=cell_info['bias'], padding_mode=cell_info['padding_mode'])
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.normalization = Normalization(cell_info['normalization'], self.output_size)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input):
        _triple = ntuple(3)
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2,
                                (self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return self.activation(self.normalization(F.conv3d(F.pad(input, expanded_padding, mode='circular'),
                                                               self.weight, self.bias, self.stride, _triple(0),
                                                               self.dilation, self.groups)))
        return self.activation(self.normalization(F.conv3d(input, self.weight, self.bias, self.stride,
                                                           self.padding, self.dilation, self.groups)))

    def extra_repr(self):
        return str(self.cell_info)


class ConvTranspose3dCell(nn.ConvTranspose3d):
    def __init__(self, cell_info):
        default_cell_info = {'stride': 1, 'padding': 0, 'output_padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros'}
        cell_info = {**default_cell_info, **cell_info}
        super(ConvTranspose3dCell, self).__init__(cell_info['input_size'], cell_info['output_size'],
                                                  cell_info['kernel_size'],
                                                  stride=cell_info['stride'], padding=cell_info['padding'],
                                                  output_padding=cell_info['output_padding'],
                                                  dilation=cell_info['dilation'], groups=cell_info['groups'],
                                                  bias=cell_info['bias'], padding_mode=cell_info['padding_mode'])
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.normalization = Normalization(cell_info['normalization'], self.output_size)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose3d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        return self.activation(self.normalization(F.conv_transpose3d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)))

    def extra_repr(self):
        return str(self.cell_info)


class ResConv3dCell(nn.Module):
    def __init__(self, cell_info):
        super(ResConv3dCell, self).__init__()
        default_cell_info = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True,
                             'padding_mode': 'zeros'}
        cell_info = {**default_cell_info, **cell_info}
        conv1_info = {**cell_info}
        conv2_info = {**cell_info, 'normalization': 'none', 'activation': 'none'}
        self.input_size = cell_info['input_size']
        self.output_size = cell_info['output_size']
        self.conv1 = Conv3dCell(conv1_info)
        self.conv2 = Conv3dCell(conv2_info)
        self.normalization = Normalization(cell_info['normalization'], self.output_size)
        self.activation = Activation(cell_info['activation'])

    def forward(self, input):
        identity = input
        x = self.conv1(input)
        x = self.conv2(x)
        output = self.activation(x + identity)
        return output

    def extra_repr(self):
        return str(self.cell_info)


class Shuffle3dCell(nn.Module):
    def __init__(self, cell_info):
        super(Shuffle3dCell, self).__init__()
        self.cell_info = cell_info
        if self.cell_info['mode'] == 'down':
            self.cell = UnShuffle3d(cell_info['scale'])
        elif cell_info['mode'] == 'up':
            self.cell = Shuffle3d(cell_info['scale'])
        else:
            raise ValueError('Not valid cell info')

    def forward(self, input):
        x = self.cell(input)
        return x