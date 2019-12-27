import config
import torch
import torch.nn as nn
from modules import make_cell


def make_model(model):
    if isinstance(model, dict):
        if 'cell' in model:
            return make_cell(model)
        elif 'nn' in model:
            return eval(model['nn'])
        else:
            cell = nn.ModuleDict({})
            for k in model:
                cell[k] = make_model(model[k])
            return cell
    elif isinstance(model, list):
        cell = nn.ModuleList([])
        for i in range(len(model)):
            cell.append(make_model(model[i]))
        return cell
    elif isinstance(model, tuple):
        container = []
        for i in range(len(model)):
            container.append(make_model(model[i]))
        cell = nn.Sequential(*container)
        return cell
    else:
        raise ValueError('Not valid model format')
    return


def normalize(input):
    broadcast_size = [1] * input.dim()
    broadcast_size[1] = input.size(1)
    m = config.PARAM['stats'].mean.view(broadcast_size).to(input.device)
    s = config.PARAM['stats'].std.view(broadcast_size).to(input.device)
    input = input.sub(m).div(s)
    return input


def denormalize(input):
    broadcast_size = [1] * input.dim()
    broadcast_size[1] = input.size(1)
    m = config.PARAM['stats'].mean.view(broadcast_size).to(input.device)
    s = config.PARAM['stats'].std.view(broadcast_size).to(input.device)
    input = input.mul(s).add(m)
    return input


