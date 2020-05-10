import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_model


def loss(input, output):
    MSE = F.mse_loss(output['H'], input['H'], reduction='mean')
    return MSE


class Cascade(nn.Module):
    def __init__(self):
        super(Cascade, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['Phy']
        encoder_cascaded = []
        for i in range(len(self.model['encoder_cascader'])):
            encoder_cascaded.append(self.model['encoder_cascader'][i](x))
        residual = input['Phy'] - sum(encoder_cascaded)
        encoder_cascaded.append(residual)
        num_node = len(encoder_cascaded)
        encoder_cascaded = torch.cat(encoder_cascaded, dim=0)
        encoded = self.model['encoder'](encoder_cascaded)
        decoded = self.model['decoder'](encoded)
        decoded = torch.chunk(decoded, num_node, dim=0)
        decoder_cascaded = []
        for i in range(len(self.model['decoder_cascader'])):
            decoder_cascaded.append(self.model['decoder_cascader'][i](decoded[i]))
        decoder_cascaded.append(decoded[-1])
        output['Phy'] = sum(decoder_cascaded)
        output['loss'] = F.mse_loss(output['Phy'], input['Phy'], reduction='mean')
        return output


def cascade():
    normalization = config.PARAM['normalization']
    activation = config.PARAM['activation']
    hidden_size = config.PARAM['hidden_size']
    cascade_size = config.PARAM['cascade_size']
    depth = config.PARAM['depth']
    config.PARAM['model'] = {}
    # Cascader
    config.PARAM['model']['encoder_cascader'] = []
    input_size = 3
    output_size = 3
    for i in range(cascade_size):
        kernel_size = 2 * i + 1
        padding = i
        config.PARAM['model']['encoder_cascader'].append(
            {'cell': 'Conv3dCell', 'input_size': input_size, 'output_size': output_size, 'kernel_size': kernel_size,
             'stride': 1, 'padding': padding, 'bias': True, 'normalization': 'none', 'activation': 'none'})
    config.PARAM['model']['decoder_cascader'] = []
    for i in range(cascade_size):
        kernel_size = 2 * i + 1
        padding = i
        config.PARAM['model']['decoder_cascader'].append(
            {'cell': 'Conv3dCell', 'input_size': input_size, 'output_size': output_size, 'kernel_size': kernel_size,
             'stride': 1, 'padding': padding, 'bias': True, 'normalization': 'none', 'activation': 'none'})
    # Encoder
    config.PARAM['model']['encoder'] = []
    input_size = 3
    output_size = hidden_size
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv3dCell', 'input_size': input_size, 'output_size': output_size, 'normalization': normalization,
         'activation': activation})
    for i in range(depth):
        input_size = hidden_size
        output_size = hidden_size
        config.PARAM['model']['encoder'].append(
            {'cell': 'ResConv3dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
             'normalization': normalization, 'activation': activation, 'mode': 'down'})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    # Decoder
    config.PARAM['model']['decoder'] = []
    input_size = hidden_size
    output_size = hidden_size
    for i in range(depth):
        config.PARAM['model']['decoder'].append(
            {'cell': 'ResConv3dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
             'normalization': normalization, 'activation': activation, 'mode': 'up'})
    input_size = hidden_size
    output_size = 3
    config.PARAM['model']['decoder'].append(
        {'cell': 'Conv3dCell', 'input_size': input_size, 'output_size': output_size, 'normalization': normalization,
         'activation': activation})
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = Cascade()
    return model