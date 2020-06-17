import torch
import torch.nn.functional as F
from config import cfg
from utils import recur


def MSE(output, target):
    with torch.no_grad():
        mse = F.mse_loss(output, target, reduction='mean').item()
    return mse


class Metric(object):
    def __init__(self):
        self.metric = {}
        self.metric['Loss'] = lambda input, output: output['loss'].item()
        self.metric['MSE'] = lambda input, output: recur(MSE, output[cfg['subset']], input[cfg['subset']])

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation