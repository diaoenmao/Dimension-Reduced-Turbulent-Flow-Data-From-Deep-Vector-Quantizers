import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

def init_param(m):
    if isinstance(m, (nn.BatchNorm3d)):
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


def physics_old(A):
    A11, A22, A33 = A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]
    continuity = (A11 + A22 + A33).mean()
    S = 0.5 * (A + A.transpose(1, 2))
    R = 0.5 * (A - A.transpose(1, 2))
    S_ijS_ij = (S * S).sum(dim=[1, 2])
    R_ijR_ij = (R * R).sum(dim=[1, 2])
    flow = (S_ijS_ij - R_ijR_ij).mean()
    
    S=S.permute(0,3,4,5,1,2)
    R=R.permute(0,3,4,5,1,2)
    SijSkjSji = torch.sum(torch.matmul(S, S) * S, axis=(4, 5))
    Omega= torch.empty((*R.size()[:-1],1), device =R.device)
    Omega[:, :, :, :, 0, 0]=2*R[:, :, :, :, 2, 1]
    Omega[:, :, :, :, 1, 0]=2*R[:, :, :, :, 0, 2]
    Omega[:, :, :, :, 2, 0]=2*R[:, :, :, :, 1, 0]
    VS_3d=torch.matmul(S,Omega)
    VortexStret=torch.matmul(Omega.transpose(4,5),VS_3d)
    Betchov2 = (SijSkjSji.mean() + (3/4)* VortexStret.mean())
    output = continuity + flow +Betchov2
    return output


def physics(A_model, A_target):
    #continuity = [None, None]
    S_ijS_ij_m = [None, None]
    R_ijR_ij_m = [None, None]
    SijSkjSji_m = [None, None]
    VortexStret_m = [None, None]
    for i, A in enumerate([A_model, A_target]):        
        A11, A22, A33 = A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]
        #continuity[i] = (A11 + A22 + A33).mean()
        S = 0.5 * (A + A.transpose(1, 2))
        R = 0.5 * (A - A.transpose(1, 2))
        S_ijS_ij = (S * S).sum(dim=[1, 2])
        R_ijR_ij = (R * R).sum(dim=[1, 2])
        S_ijS_ij_m[i] = S_ijS_ij.mean()
        R_ijR_ij_m[i] = R_ijR_ij.mean()
        
        S=S.permute(0,3,4,5,1,2).reshape(-1,3,3)
        R=R.permute(0,3,4,5,1,2).reshape(-1,3,3)         
        SijSkjSji = torch.sum(torch.matmul(S, S) * S, axis=(1, 2))
        Omega= torch.empty((*R.size()[:-1],1), device =R.device)
        Omega[:, 0, 0]=2*R[:, 2, 1]
        Omega[:, 1, 0]=2*R[:, 0, 2]
        Omega[:, 2, 0]=2*R[:, 1, 0]
        VS_3d=torch.matmul(S,Omega)
        VortexStret=torch.matmul(Omega.transpose(1,2),VS_3d)        
        SijSkjSji_m[i] = SijSkjSji.mean()
        VortexStret_m[i] = (-3/4) * VortexStret.mean()    
    
    weight = torch.tensor([1, 1, 1, 1], device=A_model.device)
    output = 0
    for i, item in enumerate([S_ijS_ij_m, R_ijR_ij_m, SijSkjSji_m, VortexStret_m]):
        output += (item[1] - item[0]).abs() * weight[i] 
        
    return output


# def weighted_mse_loss(input, target, weight=(2. * torch.ones(3, 3)).fill_diagonal_(1)):
#     weight = weight.view(3, 3, 1, 1, 1).to(input.device)
#     loss = nn.functional.mse_loss(input, target, reduction='none')
#     loss = (loss * weight).mean()
#     return loss

def weighted_mse_loss(input, target, weight=(2.*torch.ones(3,3)).fill_diagonal_(1)):
    loss_=nn.functional.mse_loss(input, target, reduction='none')
    return sum([(weight[i,j]*loss_[:,i,j]).mean() for i in range(3) for j in range(3)])


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window


def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    L = torch.max(torch.max(img1), torch.max(img2)) -  torch.min(torch.min(img1), torch.min(img2))
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

def ssim3D(img1, img2, window_size = 8, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)
