import numpy as np
import torch
import torch.nn as nn


def init_param(m):
    if isinstance(m, (nn.BatchNorm3d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    return m


def spectral_derivative_3d(V):
    N, C, H, W, D = V.size()
    h = np.fft.fftfreq(H, 1. / H)
    w = np.fft.fftfreq(W, 1. / W)
    d = np.fft.fftfreq(D, 1. / D)
    mesh_h, mesh_w, mesh_d = torch.tensor(np.meshgrid(h, w, d, indexing='ij'), device=V.device, dtype=V.dtype)
    V_fft = torch.rfft(V, signal_ndim=3, onesided=False)
    V_fft_hat = torch.stack([-V_fft[..., 1], V_fft[..., 0]], dim=-1)
    dV_dh = torch.irfft(V_fft_hat * mesh_h.unsqueeze(-1), signal_ndim=3, onesided=False)
    dV_dw = torch.irfft(V_fft_hat * mesh_w.unsqueeze(-1), signal_ndim=3, onesided=False)
    dV_dd = torch.irfft(V_fft_hat * mesh_d.unsqueeze(-1), signal_ndim=3, onesided=False)
    dV = torch.stack([dV_dh, dV_dw, dV_dd], dim=2)
    return dV