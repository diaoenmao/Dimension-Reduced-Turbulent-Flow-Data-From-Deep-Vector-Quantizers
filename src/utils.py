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
    cfg['depth'] = int(cfg['control']['depth'])
    cfg['d_commit'] = float(cfg['control']['d_commit'])
    if cfg['data_name'] in ['Turb']:
        cfg['data_shape'] = [3, 128, 128, 128]
    if cfg['data_name'] in ['Turb']:
        cfg['vqvae'] = {}
        cfg['vqvae']['hidden_size'] = 128
        cfg['vqvae']['num_res_block'] = 2
        cfg['vqvae']['res_size'] = 32
        cfg['vqvae']['embedding_size'] = 64
        cfg['vqvae']['num_embedding'] = 512
        cfg['vqvae']['vq_commit'] = 0.25
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


def Q_R(A):
    A11, A12, A13 = A[:, 0, 0], A[:, 0, 1], A[:, 0, 2]
    A21, A22, A23 = A[:, 1, 0], A[:, 1, 1], A[:, 1, 2]
    A31, A32, A33 = A[:, 2, 0], A[:, 2, 1], A[:, 2, 2]
    trace_A1 = A11 + A22 + A33
    trace_A2 = A11 ** 2 + A22 ** 2 + A33 ** 2 + 2 * (A12 * A21 + A13 * A31 + A23 * A32)
    trace_A3 = A11 * (A11 ** 2 + A12 * A21 + A13 * A31) + \
               A22 * (A22 ** 2 + A12 * A21 + A23 * A32) + \
               A33 * (A33 ** 2 + A13 * A31 + A23 * A32) + \
               A21 * (A11 * A12 + A12 * A22 + A13 * A32) + \
               A31 * (A11 * A13 + A12 * A23 + A13 * A33) + \
               A12 * (A11 * A21 + A21 * A22 + A23 * A31) + \
               A32 * (A13 * A21 + A22 * A23 + A23 * A33) + \
               A13 * (A11 * A31 + A21 * A32 + A31 * A33) + \
               A23 * (A12 * A31 + A22 * A32 + A32 * A33)
    Q = (-1 / 2) * trace_A2
    R = (-1 / 3) * trace_A3
    S_ijS_ij = ((1 / 2) * (A11 + A11)) ** 2 + ((1 / 2) * (A22 + A22)) ** 2 + ((1 / 2) * (A33 + A33)) ** 2 + \
               2 * ((1 / 2) * (A12 + A21)) ** 2 + 2 * ((1 / 2) * (A13 + A31)) ** 2 + 2 * (
                       (1 / 2) * (A23 + A32)) ** 2
    R_ijR_ij = 2 * ((1 / 2) * (A12 - A21)) ** 2 + 2 * ((1 / 2) * (A13 - A31)) ** 2 + 2 * (
            (1 / 2) * (A23 - A32)) ** 2
    return Q, R, S_ijS_ij, R_ijR_ij


def plot_PDF_VelocityGrad_DL_Model_DNS(A11_DNS, A12_DNS, A13_DNS, A21_DNS, A22_DNS, A23_DNS, A31_DNS, A32_DNS, A33_DNS,
                                       A11_Model, A12_Model, A13_Model, A21_Model, A22_Model, A23_Model, A31_Model,
                                       A32_Model, A33_Model, str_list_var=None, num_bins=int(1000 // 2), path=None):
    if str_list_var is None:
        str_list_var = ['dUdx_Phy', 'dUdy_Phy', 'dUdz_Phy', 'dVdx_Phy', 'dVdy_Phy', 'dVdz_Phy', 'dWdx_Phy', 'dWdy_Phy',
                        'dWdz_Phy']
    import scipy.stats as stats
    def replaceZeroes(data):
        min_nonzero = np.min(data[np.nonzero(data)])
        data[data == 0] = min_nonzero
        return data

    list_var_Model = [A11_Model, A12_Model, A13_Model, A21_Model, A22_Model, A23_Model, A31_Model, A32_Model, A33_Model]
    list_var_DNS = [A11_DNS, A12_DNS, A13_DNS, A21_DNS, A22_DNS, A23_DNS, A31_DNS, A32_DNS, A33_DNS]
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 25))
    fontsize = 15
    for i, var_Model, var_DNS, var_name in zip(range(9), list_var_Model, list_var_DNS, str_list_var):
        # Model
        p, x = np.histogram((var_Model.ravel() - np.mean(var_Model.ravel())) / np.std(var_Model.ravel()), density=True,
                            bins=num_bins)
        x = x[:-1] + (x[1] - x[0]) / 2
        y = np.log10(replaceZeroes(p))
        axes[i // 3][i % 3].plot(x, y, 'b', lw=2, label=var_name + '_Model')
        # DNS
        p, x = np.histogram((var_DNS.ravel() - np.mean(var_DNS.ravel())) / np.std(var_DNS.ravel()), density=True,
                            bins=num_bins)
        x = x[:-1] + (x[1] - x[0]) / 2
        y = np.log10(replaceZeroes(p))
        axes[i // 3][i % 3].plot(x, y, 'g', lw=2, label=var_name + '_DNS')
        axes[i // 3][i % 3].set_title("MSE = %.4f" % np.mean((var_DNS - var_Model) ** 2), fontsize=fontsize)
        axes[i // 3][i % 3].set_xlim(-10, 10)
        axes[i // 3][i % 3].set_ylim(-5, 0)
        axes[i // 3][i % 3].set_xlabel("Normalized " + var_name, fontsize=fontsize)
        axes[i // 3][i % 3].set_ylabel('log10(PDF)', fontsize=fontsize)
        axes[i // 3][i % 3].grid(True)
        xx = np.linspace(-5, 5, 1000)
        axes[i // 3][i % 3].plot(xx, np.log10(stats.norm.pdf(xx, 0, 1)), 'r--', label="Gaussian")
        axes[i // 3][i % 3].legend(fontsize=fontsize)
    if path is not None:
        plt.tight_layout()
        makedir_exist_ok(path)
        fig.savefig('{}/vg_{}.{}'.format(path, cfg['model_tag'], cfg['fig_format']), dpi=300, bbox_inches='tight',
                    fontsize=fontsize)
        plt.close()


def vis(input, output, path, i_d_min=5, fontsize=10, num_bins=1500):
    input_uvw = input['uvw'].cpu().numpy()
    input_duvw = input['duvw'].cpu().numpy()
    output_uvw = output['uvw'].cpu().numpy()
    output_duvw = output['duvw'].cpu().numpy()
    import scipy.stats as stats
    j_d_min, j_d_max = 0, 128
    k_d_min, k_d_max = 0, 128
    label = ['U', 'V', 'W']
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    for i in range(3):
        plt.colorbar(ax[i][0].imshow(input_uvw[0, i, i_d_min:(i_d_min + 1), j_d_min:j_d_max,
                                     k_d_min:k_d_max].squeeze()), ax=ax[i][0], fraction=0.046, pad=0.04)
        plt.colorbar(ax[i][1].imshow(output_uvw[0, i, i_d_min:(i_d_min + 1), j_d_min:j_d_max,
                                     k_d_min:k_d_max].squeeze()), ax=ax[i][1], fraction=0.046, pad=0.04)
        ax[i][0].set_title('Original {}'.format(label[i]), fontsize=fontsize)
        ax[i][1].set_title('Reconstructed {}'.format(label[i]), fontsize=fontsize)
        p, x = np.histogram((input_uvw[0, i, :, :, :].ravel() - np.mean(input_uvw[0, i, :, :, :].ravel()))
                            / np.std(input_uvw[0, i, :, :, :].ravel()), density=True, bins=num_bins)
        x = x[:-1] + (x[1] - x[0]) / 2
        p[p == 0] = np.min(p[np.nonzero(p)])
        y = np.log10(p)
        ax[i][2].plot(x, y, 'b', lw=2, label='Original {}'.format(label[i]))
        p, x = np.histogram((output_uvw[:, i, :, :, :].ravel() - np.mean(output_uvw[:, i, :, :, :].ravel()))
                            / np.std(output_uvw[:, i, :, :, :].ravel()), density=True, bins=num_bins)
        x = x[:-1] + (x[1] - x[0]) / 2
        p[p == 0] = np.min(p[np.nonzero(p)])
        y = np.log10(p)
        ax[i][2].plot(x, y, 'g', lw=2, label='Reconstructed {}'.format(label[i]))
        ax[i][2].set_xlim(-10, 10)
        ax[i][2].set_ylim(-5, 0)
        ax[i][2].set_xlabel('Normalized {}'.format(label[i]), fontsize=fontsize)
        ax[i][2].set_ylabel('log10(pdf)', fontsize=fontsize)
        ax[i][2].set_title('MSE = {:.4f}'.format(np.mean((output_uvw[:, i, :, :, :] - input_uvw[:, i, :, :, :]) ** 2)),
                           fontsize=fontsize)
        ax[i][2].grid(True)
        xx = np.linspace(-5, 5, 1000)
        ax[i][2].plot(xx, np.log10(stats.norm.pdf(xx, 0, 1)), 'r--', label="Gaussian")
        ax[i][2].legend(fontsize=fontsize)
    plt.tight_layout()
    makedir_exist_ok(path)
    fig.savefig('{}/uvw_{}.{}'.format(path, cfg['model_tag'], cfg['fig_format']), dpi=300, bbox_inches='tight',
                fontsize=fontsize)
    plt.close()
    label = [['dUdx', 'dUdy', 'dUdz'], ['dVdx', 'dVdy', 'dVdz'], ['dWdx', 'dWdy', 'dWdz']]
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 25))
    fontsize = 15
    for i in range(3):
        for j in range(3):
            p, x = np.histogram(
                (input_duvw[:, i, j, :, :, :].ravel() - np.mean(input_duvw[:, i, j, :, :, :].ravel())) / np.std(
                    input_duvw[:, i, j, :, :, :].ravel()), density=True, bins=num_bins)
            x = x[:-1] + (x[1] - x[0]) / 2
            p[p == 0] = np.min(p[np.nonzero(p)])
            y = np.log10(p)
            ax[i][j].plot(x, y, 'g', lw=2, label='Original {}'.format(label[i][j]))
            p, x = np.histogram(
                (output_duvw[:, i, j, :, :, :].ravel() - np.mean(output_duvw[:, i, j, :, :, :].ravel())) / np.std(
                    output_duvw[:, i, j, :, :, :].ravel()), density=True, bins=num_bins)
            x = x[:-1] + (x[1] - x[0]) / 2
            p[p == 0] = np.min(p[np.nonzero(p)])
            y = np.log10(p)
            ax[i][j].plot(x, y, 'b', lw=2, label='Reconstructed {}'.format(label[i][j]))
            ax[i][j].set_title('MSE = {:.4f}'.format(np.mean((output_duvw[:, i, :, :, :] -
                                                              input_duvw[:, i, :, :, :]) ** 2)), fontsize=fontsize)
            ax[i][j].set_xlim(-10, 10)
            ax[i][j].set_ylim(-5, 0)
            ax[i][j].set_xlabel('Normalized {}'.format(label[i][j]), fontsize=fontsize)
            ax[i][j].set_ylabel('log10(PDF)', fontsize=fontsize)
            ax[i][j].grid(True)
            xx = np.linspace(-5, 5, 1000)
            ax[i][j].plot(xx, np.log10(stats.norm.pdf(xx, 0, 1)), 'r--', label="Gaussian")
            ax[i][j].legend(fontsize=fontsize)
    plt.tight_layout()
    makedir_exist_ok(path)
    fig.savefig('{}/vg_{}.{}'.format(path, cfg['model_tag'], cfg['fig_format']), dpi=300, bbox_inches='tight',
                fontsize=fontsize)
    plt.close()
    # input_Q, input_R, input_S_ijS_ij, input_R_ijR_ij = Q_R(input_duvw)
    # output_Q, output_R, output_S_ijS_ij, output_R_ijR_ij = Q_R(output_duvw)
    return