import collections.abc as container_abcs
import errno
import numpy as np
import os
import torch
from itertools import repeat
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from config import cfg


def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, protocol=2, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path, pickle_protocol=protocol)
    elif mode == 'numpy':
        np.save(path, input, allow_pickle=True)
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'numpy':
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path, nrow=10, padding=2, pad_value=0, range=None):
    makedir_exist_ok(os.path.dirname(path))
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, range=range)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    else:
        raise ValueError('Not valid input type')
    return output


def process_dataset(dataset):
    return


def process_control():
    if cfg['data_name'] in ['Turb']:
        cfg['data_shape'] = [3, 128, 128, 128]
    if cfg['data_name'] in ['Turb']:
        cfg['hidden_size'] = 128
        cfg['num_res_block'] = 2
        cfg['num_res_channel'] = 32
        cfg['embedding_dim'] = 64
        cfg['num_embedding'] = 512
        cfg['vq_commit'] = 0.25
    return


def make_stats(dataset):
    if os.path.exists('./data/stats/{}.pt'.format(dataset.data_name)):
        stats = load('./data/stats/{}.pt'.format(dataset.data_name))
    elif dataset is not None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
        stats = Stats(dim=1)
        with torch.no_grad():
            for input in data_loader:
                stats.update(input['img'])
        save(stats, './data/stats/{}.pt'.format(dataset.data_name))
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)
            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n
        return


def resume(model, model_tag, optimizer=None, scheduler=None, load_tag='checkpoint', verbose=True):
    if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
        checkpoint = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
        logger = checkpoint['logger']
        if verbose:
            print('Resume from {}'.format(last_epoch))
    else:
        print('Not exists model tag: {}, start from scratch'.format(model_tag))
        from datetime import datetime
        from logger import Logger
        last_epoch = 1
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
    return last_epoch, model, optimizer, scheduler, logger


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input


def vis(signal, recon_signal, path, i_d_min=5, fontsize=10, num_bins=1500):
    import scipy.stats as stats
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    j_d_min, j_d_max = 0, 128
    k_d_min, k_d_max = 0, 128
    label = ['U', 'V', 'W']
    for i in range(3):
        plt.colorbar(ax[i][0].imshow(signal[0, i, i_d_min:(i_d_min + 1), j_d_min:j_d_max,
                                     k_d_min:k_d_max].squeeze()), ax=ax[i][0], fraction=0.046, pad=0.04)
        plt.colorbar(ax[i][1].imshow(recon_signal[0, i, i_d_min:(i_d_min + 1), j_d_min:j_d_max,
                                     k_d_min:k_d_max].squeeze()), ax=ax[i][1], fraction=0.046, pad=0.04)
        ax[i][0].set_title('Original {}'.format(label[i]), fontsize=fontsize)
        ax[i][1].set_title('Reconstructed {}'.format(label[i]), fontsize=fontsize)

        p, x = np.histogram((signal.ravel() - np.mean(signal.ravel())) / np.std(signal.ravel()), density=True,
                            bins=num_bins)
        x = x[:-1] + (x[1] - x[0]) / 2
        p[p == 0] = np.min(p[np.nonzero(p)])
        y = np.log10(p)
        ax[i][2].plot(x, y, 'b', lw=2, label='Original {}'.format(label[i]))
        p, x = np.histogram((recon_signal.ravel() - np.mean(recon_signal.ravel())) / np.std(recon_signal.ravel()),
                            density=True, bins=num_bins)
        x = x[:-1] + (x[1] - x[0]) / 2
        p[p == 0] = np.min(p[np.nonzero(p)])
        y = np.log10(p)
        ax[i][2].plot(x, y, 'g', lw=2, label='Reconstructed {}'.format(label[i]))
        ax[i][2].set_xlim(-6, 6)
        ax[i][2].set_ylim(-5, 0)
        ax[i][2].set_xlabel('Normalized Signal', fontsize=fontsize)
        ax[i][2].set_ylabel('log10(pdf)', fontsize=fontsize)
        ax[i][2].grid(True)
        xx = np.linspace(-5, 5, 1000)
        ax[i][2].plot(xx, np.log10(stats.norm.pdf(xx, 0, 1)), 'r--', label="Gaussian")
        ax[i][2].legend(fontsize=fontsize)
    plt.tight_layout()
    dir = os.path.dirname(path)
    makedir_exist_ok(dir)
    fig.savefig(path, dpi=300, bbox_inches='tight', fontsize=fontsize)
    plt.close()
    return