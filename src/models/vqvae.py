import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import make_model, init_param


class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        self.model = make_model(cfg['model'])

    def encode(self, input):
        x = input['img']
        x = self.model['encoder'](x)
        _, _, code = self.model['quantizer'](x)
        return code

    def decode(self, code):
        x = self.model['quantizer'].embedding_code(code).permute(0, 4, 1, 2, 3).contiguous()
        generated = self.model['decoder'](x)
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['uvw']
        x = self.model['encoder'](x)
        x, vq_loss, output['idx'] = self.model['quantizer'](x)
        decoded = self.model['decoder'](x)
        output['uvw'] = decoded
        output['loss'] = F.mse_loss(decoded, input['uvw']) + cfg['vq_commit'] * vq_loss
        return output


def vqvae():
    normalization = 'bn'
    activation = 'relu'
    data_shape = cfg['data_shape']
    hidden_size = cfg['hidden_size']
    quantizer_embedding_size = cfg['quantizer_embedding_size']
    num_embedding = cfg['num_embedding']
    cfg['model'] = {}
    # Encoder
    encoder_size = [data_shape[0], *hidden_size]
    cfg['model']['encoder'] = []
    for i in range(len(hidden_size)):
        input_size = encoder_size[i]
        output_size = encoder_size[i + 1]
        cfg['model']['encoder'].append(
            {'cell': 'Conv3dCell', 'input_size': input_size, 'output_size': output_size,
             'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': normalization,
             'activation': activation})
    input_size = encoder_size[-1]
    output_size = encoder_size[-1]
    for i in range(2):
        cfg['model']['encoder'].append(
            {'cell': 'ResConv3dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
             'normalization': normalization, 'activation': activation})
    input_size = encoder_size[-1]
    output_size = quantizer_embedding_size
    cfg['model']['encoder'].append(
        {'cell': 'Conv3dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': 'none',
         'activation': 'none'})
    cfg['model']['encoder'] = tuple(cfg['model']['encoder'])
    # Quantizer
    cfg['model']['quantizer'] = {
        'cell': 'VectorQuantization3dCell', 'embedding_dim': quantizer_embedding_size, 'num_embedding': num_embedding}
    # Decoder
    decoder_size = [*hidden_size[::-1], data_shape[0]]
    cfg['model']['decoder'] = []
    input_size = quantizer_embedding_size
    output_size = decoder_size[0]
    cfg['model']['decoder'].append(
        {'cell': 'Conv3dCell', 'input_size': input_size, 'output_size': output_size,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True, 'normalization': normalization,
         'activation': activation})
    input_size = decoder_size[0]
    output_size = decoder_size[0]
    for i in range(2):
        cfg['model']['decoder'].append(
            {'cell': 'ResConv3dCell', 'input_size': input_size, 'output_size': output_size, 'hidden_size': output_size,
             'normalization': normalization, 'activation': activation})
    for i in range(len(hidden_size)):
        input_size = decoder_size[i]
        output_size = decoder_size[i + 1]
        cfg['model']['decoder'].append(
            {'cell': 'ConvTranspose3dCell', 'input_size': input_size, 'output_size': output_size,
             'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True, 'normalization': 'none',
             'activation': 'none'})
    cfg['model']['decoder'] = tuple(cfg['model']['decoder'])
    model = VQVAE()
    model.apply(init_param)
    return model