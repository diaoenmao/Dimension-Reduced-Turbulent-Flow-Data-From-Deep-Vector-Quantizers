import torch
import torch.nn.functional as F
from config import cfg
from utils import recur
from models.utils import physics, ssim3D


def MSE(output, target):
    with torch.no_grad():
        mse = F.mse_loss(output, target, reduction='mean').item()
    return mse


def Physics(output):
    phy = physics(output).item()
    return phy


def PSNR(output, target):
    with torch.no_grad():
        mse = F.mse_loss(output, target, reduction='mean')
        max_I = torch.max(torch.max(target), torch.max(output))
    return 20 * (torch.log10(max_I / (torch.sqrt(mse)))).item()


def MAE(output, target):
    with torch.no_grad():
        mae = F.l1_loss(output, target, reduction='mean').item()
    return mae


def MSSIM(output, target):
    with torch.no_grad():
        mssim = ssim3D(output, target).item()
    return mssim


class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {'Loss': lambda input, output: output['loss'].item(),
                       'MSE': lambda input, output: recur(MSE, output['uvw'], input['uvw']),
                       'D_MSE': lambda input, output: recur(MSE, output['duvw'], input['duvw']),
                       'Physics': lambda input, output: recur(Physics, output['duvw']),
                       'PSNR': lambda input, output: recur(PSNR, output['uvw'], input['uvw']),
                       'MAE': lambda input, output: recur(MAE, output['uvw'], input['uvw']),
                       'MSSIM': lambda input, output: recur(MSSIM, output['uvw'], input['uvw'])}

    def make_metric_name(self, metric_name):
        return metric_name

    def make_pivot(self):
        if cfg['data_name'] in ['Turb']:
            if cfg['model_name'] in ['transformer', 'convlstm', 'vqvae']:
                pivot = float('inf')
                pivot_name = 'MSE'
                pivot_direction = 'down'
            else:
                raise ValueError('Not valid model name')
        else:
            raise ValueError('Not valid data name')
        return pivot, pivot_name, pivot_direction

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return
