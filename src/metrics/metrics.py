import torch
import torch.nn.functional as F
from config import cfg
from utils import recur
from models.utils import physics, weighted_mse_loss, ssim3D


def MSE(output, target):
    with torch.no_grad():
        mse = F.mse_loss(output, target, reduction='mean').item()
    return mse


def Physics(output, target):
    phy = physics(output, target).item()
    return phy


def PSNR(output, target):
    with torch.no_grad():
        mse = F.mse_loss(output, target, reduction='mean')
        max_I=torch.max(torch.max(target), torch.max(output))
    return 20*(torch.log10(max_I / (torch.sqrt(mse)) )).item()


def MAE(output, target):
    with torch.no_grad():
        mae = F.l1_loss(output, target, reduction='mean').item()
    return mae


def MSSIM(output, target):
    with torch.no_grad():
        mssim = ssim3D(output, target).item()
    return mssim



class Metric(object):
    def __init__(self):
        self.metric = {}
        self.metric['Loss'] = lambda input, output: output['loss'].item()
        self.metric['MSE'] = lambda input, output: recur(MSE, output[cfg['subset']], input[cfg['subset']])
        self.metric['D_MSE'] = lambda input, output: recur(MSE, output['d{}'.format(cfg['subset'])],
                                                           input['d{}'.format(cfg['subset'])])
        self.metric['Physics'] = lambda input, output: recur(Physics, output['d{}'.format(cfg['subset'])],
                                                            input['d{}'.format(cfg['subset'])])
        self.metric['PSNR'] = lambda input, output: recur(PSNR, output[cfg['subset']], input[cfg['subset']])
        self.metric['MAE'] = lambda input, output: recur(MAE, output[cfg['subset']], input[cfg['subset']])
        self.metric['MSSIM'] = lambda input, output: recur(MSSIM, output[cfg['subset']], input[cfg['subset']])

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation