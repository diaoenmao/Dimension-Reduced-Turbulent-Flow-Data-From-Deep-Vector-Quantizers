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


# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import kornia
# import math
# import scipy
# from utils import ntuple
#
#
# def FFT_derivative(V):
#     V = V.cpu().numpy()
#     N, C, H, W, D = V.shape
#     h = np.fft.fftfreq(H, 1. / H)
#     w = np.fft.fftfreq(W, 1. / W)
#     d = np.fft.fftfreq(D, 1. / D)
#     dV = []
#     for i in range(N):
#         V_i = V[i]
#         dV_i = []
#         for c in range(C):
#             V_i_c = V_i[c]
#             V_i_c_fft = np.fft.fftn(V_i_c)
#             mesh_h, mesh_w, mesh_d = np.meshgrid(h, w, d, indexing='ij')
#             dV_dh = np.real(np.fft.ifftn(1.0j * V_i_c_fft * mesh_h))
#             dV_dw = np.real(np.fft.ifftn(1.0j * V_i_c_fft * mesh_w))
#             dV_dd = np.real(np.fft.ifftn(1.0j * V_i_c_fft * mesh_d))
#             dV_i_c = np.stack([dV_dd, dV_dw, dV_dh], axis=0)
#             dV_i.append(dV_i_c)
#         dV_i = np.stack(dV_i, axis=0)
#         dV.append(dV_i)
#     dV = np.stack(dV, axis=0)
#     dV = torch.tensor(dV, dtype=torch.float32)
#     return dV
#
#
# def Kornia_derivative(V, dx):
#     sg = kornia.filters.SpatialGradient3d(mode='diff', order=1)
#     dV = sg(V) / dx
#     return dV
#
#
# def make_kernel(mode, d):
#     if mode == 'diff':
#         hxyz = torch.tensor([0, 1, 0], dtype=torch.float32)
#     elif mode == 'sobel':
#         hxyz = torch.tensor([1, 2, 1], dtype=torch.float32)
#     elif mode == 'scharr':
#         hxyz = torch.tensor([3, 10, 3], dtype=torch.float32)
#     else:
#         raise ValueError('Not valid mode')
#     hpxyz = torch.tensor([-1, 0, 1], dtype=torch.float32)
#     if d == 1:
#         kernel = torch.zeros((1, 3), dtype=torch.float32)
#         for i in range(3):
#             kernel[0][i] = hpxyz[i]
#         kernel = kernel / kernel.abs().sum(dim=[-1]).view(-1, 1)
#     elif d == 2:
#         kernel = torch.zeros((2, 3, 3), dtype=torch.float32)
#         for i in range(3):
#             for j in range(3):
#                 kernel[0][i][j] = hpxyz[i] * hxyz[j]
#                 kernel[1][i][j] = hxyz[i] * hpxyz[j]
#         kernel = kernel / kernel.abs().sum(dim=[-2, -1]).view(-1, 1, 1)
#     elif d == 3:
#         kernel = torch.zeros((3, 3, 3, 3), dtype=torch.float32)
#         for i in range(3):
#             for j in range(3):
#                 for k in range(3):
#                     kernel[0][i][j][k] = hpxyz[i] * hxyz[j] * hxyz[k]
#                     kernel[1][i][j][k] = hxyz[i] * hpxyz[j] * hxyz[k]
#                     kernel[2][i][j][k] = hxyz[i] * hxyz[j] * hpxyz[k]
#         kernel = kernel / kernel.abs().sum(dim=[-3, -2, -1]).view(-1, 1, 1, 1)
#     kernel = kernel.flip(0)
#     return kernel
#
#
# class SpatialGradient3d(nn.Module):
#     def __init__(self, mode):
#         super().__init__()
#         self.mode = mode
#         self.register_buffer('kernel', make_kernel(self.mode, 3))
#
#     def forward(self, input):
#         B, C, H, W, D = input.size()
#         kernel = self.kernel.unsqueeze(1).repeat(C, 1, 1, 1, 1)
#         padding = tuple(x for x in reversed(ntuple(3)(kernel.size(-1) // 2)) for _ in range(2))
#         output = F.conv3d(F.pad(input, padding, 'circular'), kernel, padding=0, groups=C).view(B, C, 3, H, W, D)
#         return output
#
#
# def Finite_derivative(V, dx, mode='sobel'):
#     sg = SpatialGradient3d(mode)
#     dV = sg(V) / dx
#     return dV
#
#
# def FFT_derivative_pytorch(V):
#     N, C, H, W, D = V.size()
#     h = torch.tensor(np.fft.fftfreq(H, 1. / H), device=V.device, dtype=V.dtype)
#     w = torch.tensor(np.fft.fftfreq(W, 1. / W), device=V.device, dtype=V.dtype)
#     d = torch.tensor(np.fft.fftfreq(D, 1. / D), device=V.device, dtype=V.dtype)
#     mesh_h, mesh_w, mesh_d = torch.tensor(np.meshgrid(h, w, d, indexing='ij'), device=V.device, dtype=V.dtype)
#     V_fft = torch.rfft(V, signal_ndim=3, onesided=False)
#     V_fft_hat = torch.stack([-V_fft[..., 1], V_fft[..., 0]], dim=-1)
#     dV_dh = torch.irfft(V_fft_hat * mesh_h.unsqueeze(-1), signal_ndim=3, onesided=False)
#     dV_dw = torch.irfft(V_fft_hat * mesh_w.unsqueeze(-1), signal_ndim=3, onesided=False)
#     dV_dd = torch.irfft(V_fft_hat * mesh_d.unsqueeze(-1), signal_ndim=3, onesided=False)
#     dV = torch.stack([dV_dd, dV_dw, dV_dh], dim=2)
#     return dV
#
#
# if __name__ == "__main__":
#     process_control()
#     cfg['batch_size']['train'] = 5
#     dataset = fetch_dataset('Turb', subset='uvw')
#     data_loader = make_data_loader(dataset)
#     input = next(iter(data_loader['train']))
#     input = collate(input)
#     print(input['ts'].size(), input['uvw'].size())
#
#     V = input['uvw']
#     dx = 2 * math.pi / 128
#     FFT_dV = FFT_derivative(V)
#     # Kornia_dV = Kornia_derivative(V, dx)
#     # diff_dV = Finite_derivative(V, dx, mode='diff')
#     # sobel_dV = Finite_derivative(V, dx, mode='sobel')
#     # scharr_dV = Finite_derivative(V, dx, mode='scharr')
#     FFT_pytorch_dV = FFT_derivative_pytorch(V)
#     # print('Kornia error', F.l1_loss(FFT_dV, Kornia_dV))
#     # print('Diff error', F.l1_loss(FFT_dV, diff_dV))
#     # print('Sobel error', F.l1_loss(FFT_dV, sobel_dV))
#     # print('Scharr error', F.l1_loss(FFT_dV, scharr_dV))
#     print('FFT Pytorch error', F.l1_loss(FFT_dV, FFT_pytorch_dV))

#
# from models.utils import spectral_derivative_3d
# from utils import Q_R
#
# if __name__ == '__main__':
#     data_name = 'Turb'
#     cfg['subset'] = 'uvw'
#     cfg['batch_size']['train'] = 1
#     dataset = fetch_dataset(data_name, cfg['subset'])
#     process_dataset(dataset['train'])
#     data_loader = make_data_loader(dataset)
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         input_size = input['uvw'].size(0)
#         print(input['uvw'].size(), input['duvw'].size())
#         duvw = input['duvw']
#         myduvw = spectral_derivative_3d(input['uvw'])
#         error = (myduvw - duvw).abs().mean([0,-3,-2,-1])
#         print(error)
#         exit()


# if __name__ == '__main__':
#     data_name = 'Turb'
#     cfg['subset'] = 'uvw'
#     cfg['batch_size']['train'] = 1
#     dataset = fetch_dataset(data_name, cfg['subset'])
#     process_dataset(dataset['train'])
#     data_loader = make_data_loader(dataset)
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         input_size = input['uvw'].size(0)
#         print(input['uvw'].size(), input['duvw'].size())
#         exit()

# import torch.fft
# import torch.nn.functional as F
# from scipy.fftpack import fftn
#
# if __name__ == '__main__':
#     a = torch.randn(1, 3, 8, 8, 8)
#     b = torch.rfft(a, signal_ndim=3, onesided=False)
#     c = torch.fft.fftn(a, dim=[-3, -2,-1])
#     d = fftn(a.numpy(), axes=[-3, -2,-1])
#     print(b.size(), c.size())
#     print(torch.allclose(b[..., 0], c.real))
#     print(F.l1_loss(c.real, b[..., 0], reduction='mean'))
#     print(torch.allclose(b[...,0], torch.tensor(d.real)))
#     print(F.l1_loss(b[...,0], torch.tensor(d.real), reduction='mean'))
#     print(torch.allclose(c.real, torch.tensor(d.real)))
#     print(F.l1_loss(c.real, torch.tensor(d.real), reduction='mean'))
#
#     print(torch.allclose(b[..., 1], c.imag))
#     print(F.l1_loss(c.imag, b[..., 1], reduction='mean'))
#     print(torch.allclose(b[...,1], torch.tensor(d.imag)))
#     print(F.l1_loss(b[...,1], torch.tensor(d.imag), reduction='mean'))
#     print(torch.allclose(c.imag, torch.tensor(d.imag)))
#     print(F.l1_loss(c.imag, torch.tensor(d.imag), reduction='mean'))
