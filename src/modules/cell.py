import config
import copy
import torch
import torch.nn as nn
from .shuffle import *
from .quantization import *


def Normalization(cell_info):
    if cell_info['mode'] == 'none':
        return nn.Sequential()
    elif cell_info['mode'] == 'bn':
        return nn.BatchNorm2d(cell_info['input_size'])
    elif cell_info['mode'] == 'in':
        return nn.InstanceNorm2d(cell_info['input_size'])
    elif cell_info['mode'] == 'ln':
        return nn.LayerNorm(cell_info['input_size'])
    else:
        raise ValueError('Not valid normalization')
    return


def Activation(cell_info):
    if cell_info['mode'] == 'none':
        return nn.Sequential()
    elif cell_info['mode'] == 'tanh':
        return nn.Tanh()
    elif cell_info['mode'] == 'hardtanh':
        return nn.Hardtanh()
    elif cell_info['mode'] == 'relu':
        return nn.ReLU(inplace=True)
    elif cell_info['mode'] == 'prelu':
        return nn.PReLU()
    elif cell_info['mode'] == 'elu':
        return nn.ELU(inplace=True)
    elif cell_info['mode'] == 'selu':
        return nn.SELU(inplace=True)
    elif cell_info['mode'] == 'celu':
        return nn.CELU(inplace=True)
    elif cell_info['mode'] == 'sigmoid':
        return nn.Sigmoid()
    elif cell_info['mode'] == 'softmax':
        return nn.SoftMax()
    else:
        raise ValueError('Not valid activation')
    return


class ConvCell(nn.Module):
    def __init__(self, cell_info):
        super(ConvCell, self).__init__()
        self.cell_default_info = {'stride': 1, 'padding': 0, 'bias': False,
                                  'normalization': config.PARAM['normalization'],
                                  'activation': config.PARAM['activation']}
        self.cell_info = {**self.cell_default_info, **cell_info}
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleDict({})
        cell_main_info = "nn.Conv2d(cell_info['input_size'], cell_info['output_size'], " \
                         "kernel_size=cell_info['kernel_size'], stride=cell_info['stride'], " \
                         "padding=cell_info['padding'], bias=cell_info['bias'])"
        cell_normalization_info = {'cell': 'Normalization', 'input_size': cell_info['output_size'],
                                   'mode': cell_info['normalization']}
        cell_activation_info = {'cell': 'Activation', 'mode': cell_info['activation']}
        cell['main'] = eval(cell_main_info)
        cell['activation'] = Cell(cell_activation_info)
        cell['normalization'] = Cell(cell_normalization_info)
        return cell

    def forward(self, input):
        x = input
        x = self.cell['normalization'](x)
        x = self.cell['main'](x)
        x = self.cell['activation'](x)
        return x


class ConvTransposeCell(nn.Module):
    def __init__(self, cell_info):
        super(ConvTransposeCell, self).__init__()
        self.cell_default_info = {'stride': 1, 'padding': 0, 'bias': False,
                                  'normalization': config.PARAM['normalization'],
                                  'activation': config.PARAM['activation']}
        self.cell_info = {**self.cell_default_info, **cell_info}
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleDict({})
        cell_main_info = "nn.ConvTranspose2d(cell_info['input_size'], cell_info['output_size'], " \
                         "kernel_size=cell_info['kernel_size'], stride=cell_info['stride'], " \
                         "padding=cell_info['padding'], bias=cell_info['bias'])"
        cell_normalization_info = {'cell': 'Normalization', 'input_size': cell_info['output_size'],
                                   'mode': cell_info['normalization']}
        cell_activation_info = {'cell': 'Activation', 'mode': cell_info['activation']}
        cell['main'] = eval(cell_main_info)
        cell['activation'] = Cell(cell_activation_info)
        cell['normalization'] = Cell(cell_normalization_info)
        return cell

    def forward(self, input):
        x = input
        x = self.cell['normalization'](x)
        x = self.cell['main'](x)
        x = self.cell['activation'](x)
        return x


class ResConvCell(nn.Module):
    def __init__(self, cell_info):
        super(ResConvCell, self).__init__()
        self.cell_default_info = {'stride': 1, 'padding': 0, 'bias': True,
                                  'normalization': config.PARAM['normalization'],
                                  'activation': config.PARAM['activation']}
        self.cell_info = {**self.cell_default_info, **cell_info}
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleDict({})
        cell_main_info = {'cell': 'ConvCell',
                          'input_size': cell_info['input_size'], 'output_size': cell_info['input_size'],
                          'kernel_size': cell_info['kernel_size'], 'stride': cell_info['stride'],
                          'padding': cell_info['padding'], 'bias': cell_info['bias'],
                          'normalization': 'none', 'activation': 'none'}
        cell_normalization_info = {'cell': 'Normalization', 'input_size': cell_info['output_size'],
                                   'mode': cell_info['normalization']}
        cell_activation_info = {'cell': 'Activation', 'mode': cell_info['activation']}
        cell['main'] = nn.ModuleList([Cell(cell_main_info), Cell(cell_main_info)])
        cell['activation'] = nn.ModuleList([Cell(cell_activation_info), Cell(cell_activation_info)])
        cell['normalization'] = nn.ModuleList([Cell(cell_normalization_info), Cell(cell_normalization_info)])
        return cell

    def forward(self, input):
        x = input
        x = self.cell['normalization'][0](x)
        shortcut = x
        x = self.cell['main'][0](x)
        x = self.cell['activation'][0](x)
        x = self.cell['normalization'][1](x)
        x = self.cell['main'][1](x)
        x = self.cell['activation'][1](x + shortcut)
        return x


class ShuffleCell(nn.Module):
    def __init__(self, cell_info):
        super(ShuffleCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleDict({})
        if cell_info['mode'] == 'down':
            cell['main'] = UnShuffle2d(cell_info['scale_factor'])
        elif cell_info['mode'] == 'up':
            cell['main'] = Shuffle2d(cell_info['scale_factor'])
        else:
            raise ValueError('Not valid shufflecell')
        return cell

    def forward(self, input):
        x = input
        x = self.cell['main'](x)
        return x


class QuantizationCell(nn.Module):
    def __init__(self, cell_info):
        super(QuantizationCell, self).__init__()
        self.cell_default_info = {'ema': False, 'commitment': 1}
        self.cell_info = {**self.cell_default_info, **cell_info}
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleDict({})
        cell['main'] = Quantization(cell_info['num_embedding'], cell_info['embedding_dim'], cell_info['ema'],
                                    cell_info['commitment'])
        return cell

    def forward(self, input):
        x = input
        quantized, encoding, distances, loss = self.cell['main'](x)
        return quantized, encoding, distances, loss


class Cell(nn.Module):
    def __init__(self, cell_info):
        super(Cell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        if self.cell_info['cell'] == 'none':
            cell = nn.Sequential()
        elif self.cell_info['cell'] == 'Normalization':
            cell = Normalization(self.cell_info)
        elif self.cell_info['cell'] == 'Activation':
            cell = Activation(self.cell_info)
        elif self.cell_info['cell'] == 'ConvCell':
            cell = ConvCell(self.cell_info)
        elif self.cell_info['cell'] == 'ConvTransposeCell':
            cell = ConvTransposeCell(self.cell_info)
        elif self.cell_info['cell'] == 'ResConvCell':
            cell = ResConvCell(self.cell_info)
        elif self.cell_info['cell'] == 'ShuffleCell':
            cell = ShuffleCell(self.cell_info)
        elif self.cell_info['cell'] == 'QuantizationCell':
            cell = QuantizationCell(self.cell_info)
        else:
            raise ValueError('Not valid {} model'.format(self.cell_info['cell']))
        return cell

    def forward(self, *input):
        x = self.cell(*input)
        return x