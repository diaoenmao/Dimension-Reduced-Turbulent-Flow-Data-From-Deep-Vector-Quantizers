import config
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils import recur


def RMSE(output, target, MAX=1.0):
    with torch.no_grad():
        mse = F.mse_loss(output.to(torch.float64), target.to(torch.float64))
        rmse = torch.sqrt(mse).item()
    return rmse


class Metric(object):
    def __init__(self):
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'RMSE': (lambda input, output: recur(RMSE, output['H'], input['H']))}

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation