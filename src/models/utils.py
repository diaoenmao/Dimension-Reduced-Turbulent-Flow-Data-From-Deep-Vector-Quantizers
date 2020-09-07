import numpy as np
import torch
import torch.nn as nn


def init_param(m):
    if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
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


def physics(A):
    A11, A22, A33 = A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]
    continuity = (A11 + A22 + A33).mean()
    S = 0.5 * (A + A.transpose(1, 2))
    R = 0.5 * (A - A.transpose(1, 2))
    S_ijS_ij = (S * S).sum(dim=[1, 2])
    R_ijR_ij = (R * R).sum(dim=[1, 2])
    flow = (S_ijS_ij - R_ijR_ij).mean()
    output = continuity + flow
    return output