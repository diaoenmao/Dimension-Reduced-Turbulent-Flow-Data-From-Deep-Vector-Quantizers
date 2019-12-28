import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_model


def loss(input, output):
    MSE = F.mse_loss(output['H'], input['H'], reduction='mean')
    return MSE


class SmartTurb(nn.Module):
    def __init__(self):
        super(SmartTurb, self).__init__()
        self.model = make_model(config.PARAM['model'])

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['A']
        x = self.model['encoder'][0](x)
        encoded = []
        for i in range(1, len(self.model['encoder'])):
            x = self.model['encoder'][i](x)
            encoded.append(x)
        x = self.model['embedding'](x)
        for i in range(len(self.model['decoder']) - 1):
            x = self.model['decoder'][i](x + encoded[len(encoded) - i - 1])
        output['H'] = self.model['decoder'][-1](x)
        output['loss'] = loss(input, output)
        return output


def st():
    normalization = config.PARAM['normalization']
    activation = config.PARAM['activation']
    hidden_size = int(config.PARAM['hidden_size'])
    depth = int(config.PARAM['depth'])
    config.PARAM['model'] = {}
    # Encoder
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'Conv3dCell', 'input_size': 9, 'output_size': hidden_size, 'kernel_size': 1, 'stride': 1, 'padding': 0,
         'bias': True, 'normalization': normalization, 'activation': activation})
    for i in range(depth):
        config.PARAM['model']['encoder'].append(
            ({'cell': 'ResConv3dCell', 'input_size': hidden_size * (8 ** i), 'output_size': hidden_size * (8 ** i),
              'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
              'activation': activation},
             {'cell': 'Shuffle3dCell', 'mode': 'down', 'scale': 2}
             ))
    config.PARAM['model']['embedding'] = {
        'cell': 'ResConv3dCell', 'input_size': hidden_size * (8 ** depth), 'output_size': hidden_size * (8 ** depth),
        'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
        'activation': activation}
    # Decoder
    config.PARAM['model']['decoder'] = []
    for i in range(depth):
        config.PARAM['model']['decoder'].append(
            ({'cell': 'Shuffle3dCell', 'mode': 'up', 'scale': 2},
             {'cell': 'ResConv3dCell', 'input_size': hidden_size * (8 ** (depth - i - 1)),
              'output_size': hidden_size * (8 ** (depth - i - 1)), 'kernel_size': 3, 'stride': 1, 'padding': 1,
              'bias': True, 'normalization': normalization, 'activation': activation}
             ))
    config.PARAM['model']['decoder'].append(
        {'cell': 'Conv3dCell', 'input_size': hidden_size, 'output_size': 6, 'kernel_size': 1, 'stride': 1, 'padding': 0,
         'bias': True, 'normalization': normalization, 'activation': activation})
    model = SmartTurb()
    return model