import argparse
import datetime
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg
from data import fetch_dataset, make_data_loader
from utils import save, load, to_device, process_control, process_dataset, resume, collate
from logger import Logger

# if __name__ == "__main__":
#     process_control()
#     dataset = fetch_dataset('Turb', subset='uvw')
#     data_loader = make_data_loader(dataset)
#     input = next(iter(data_loader['train']))
#     input = collate(input)
#     print(input['ts'].size(), input['uvw'].size())
#     print(input['ts'].dtype, input['uvw'].dtype)


import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def FFT_derivative(V):
    V = V.cpu().numpy()
    N, C, H, W, D = V.shape
    h = np.fft.fftfreq(H, 1. / H)
    w = np.fft.fftfreq(W, 1. / W)
    d = np.fft.fftfreq(D, 1. / D)
    dV = []
    for i in range(N):
        V_i = V[i]
        dV_i = []
        for c in range(C):
            V_i_c = V_i[c]
            V_i_c_fft = np.fft.fftn(V_i_c)
            mesh_h, mesh_w, mesh_d = np.meshgrid(h, w, d, indexing='ij')
            dV_dh = np.real(np.fft.ifftn(1.0j * np.multiply(V_i_c_fft, mesh_h)))
            dV_dw = np.real(np.fft.ifftn(1.0j * np.multiply(V_i_c_fft, mesh_w)))
            dV_dd = np.real(np.fft.ifftn(1.0j * np.multiply(V_i_c_fft, mesh_d)))
            dV_i_c = np.stack([dV_dh, dV_dw, dV_dd], axis=0)
            dV_i.append(dV_i_c)
        dV_i = np.stack(dV_i, axis=0)
        dV.append(dV_i)
    dV = np.stack(dV, axis=0)
    dV = torch.tensor(dV)
    return dV


def Kornia_derivative(V):
    import kornia
    sg = kornia.filters.SpatialGradient3d(mode='diff', order=1)
    dV = sg(V)
    return dV


class SpatialGradient3d(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode: str = mode
        self.kernel = self.make_kernel()

    def __repr__(self):
        return '{} (mode: {})'.format(self.__class__.__name__, self.mode)

    def make_kernel(self):
        if self.mode == 'diff':
            kernel = torch.tensor([[[[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]],

                                    [[0.0, 0.0, 0.0],
                                     [-0.5, 0.0, 0.5],
                                     [0.0, 0.0, 0.0]],

                                    [[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]],
                                    ],
                                   [[[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]],

                                    [[0.0, -0.5, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.5, 0.0]],

                                    [[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]],
                                    ],
                                   [[[0.0, 0.0, 0.0],
                                     [0.0, -0.5, 0.0],
                                     [0.0, 0.0, 0.0]],

                                    [[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]],

                                    [[0.0, 0.0, 0.0],
                                     [0.0, 0.5, 0.0],
                                     [0.0, 0.0, 0.0]],
                                    ],
                                   ], dtype=torch.float32)
            kernel = kernel.unsqueeze(1)
            return kernel
        elif self.mode == 'sobel':
            hxyz = torch.tensor([1, 2, 1], dtype=torch.float32)
        elif self.mode == 'scharr':
            hxyz = torch.tensor([3, 10, 3], dtype=torch.float32)
        else:
            raise ValueError('Not valid mode')
        hpxyz = torch.tensor([1, 0, -1], dtype=torch.float32)
        kernel = torch.zeros((3, 3, 3, 3), dtype=torch.float32)
        # build kernel
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    kernel[0][i][j][k] = hpxyz[i] * hxyz[j] * hxyz[k]
                    kernel[1][i][j][k] = hxyz[i] * hpxyz[j] * hxyz[k]
                    kernel[2][i][j][k] = hxyz[i] * hxyz[j] * hpxyz[k]
        kernel = kernel.unsqueeze(1)
        kernel = kernel / kernel.abs().sum(dim=[-3,-2,-1]).view(*kernel.size()[:2], 1, 1, 1)
        return kernel

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 5:
            raise ValueError("Invalid input shape, we expect BxCxDxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, d, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype).detach()
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1, 1)

        # convolve input tensor with grad kernel
        kernel_flip: torch.Tensor = kernel.flip(-3)
        # Pad with "replicate for spatial dims, but with zeros for channel
        spatial_pad = [self.kernel.size(2) // 2,
                       self.kernel.size(2) // 2,
                       self.kernel.size(3) // 2,
                       self.kernel.size(3) // 2,
                       self.kernel.size(4) // 2,
                       self.kernel.size(4) // 2]
        out_ch = 3
        return F.conv3d(F.pad(input, spatial_pad, 'replicate'), kernel, padding=0, groups=c).view(b, c, out_ch, d, h, w)


def Finite_derivative(V, mode='sobel'):
    sg = SpatialGradient3d(mode)
    dV = sg(V)
    return dV


if __name__ == "__main__":
    process_control()
    cfg['batch_size']['train'] = 5
    dataset = fetch_dataset('Turb', subset='uvw')
    data_loader = make_data_loader(dataset)
    input = next(iter(data_loader['train']))
    input = collate(input)
    print(input['ts'].size(), input['uvw'].size())
    V = input['uvw']
    FFT_dV = FFT_derivative(V)
    Kornia_dV = Kornia_derivative(V)
    sobel_dV = Finite_derivative(V, mode='sobel')
    scharr_dV = Finite_derivative(V, mode='scharr')
    print((FFT_dV - Kornia_dV).abs().mean())
    print((FFT_dV - sobel_dV).abs().mean())
    print((FFT_dV - scharr_dV).abs().mean())