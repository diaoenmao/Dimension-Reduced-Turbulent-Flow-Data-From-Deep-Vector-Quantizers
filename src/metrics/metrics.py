import torch
import torch.nn.functional as F
from config import cfg
from utils import recur
from models.utils import physics


def MSE(output, target):
    with torch.no_grad():
        mse = F.mse_loss(output, target, reduction='mean').item()
    return mse


def Physics(output):
    phy = physics(output).item()
    return phy


class Metric(object):
    def __init__(self):
        self.metric = {}
        self.metric['Loss'] = lambda input, output: output['loss'].item()
        self.metric['MSE'] = lambda input, output: recur(MSE, output[cfg['subset']], input[cfg['subset']])
        self.metric['D_MSE'] = lambda input, output: recur(MSE, output['d{}'.format(cfg['subset'])],
                                                           input['d{}'.format(cfg['subset'])])
        self.metric['Physics'] = lambda input, output: recur(Physics, output['d{}'.format(cfg['subset'])])

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation