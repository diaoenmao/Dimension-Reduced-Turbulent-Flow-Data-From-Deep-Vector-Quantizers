import collections.abc as container_abcs
import errno
import numpy as np
import os
import torch
import torch.optim as optim
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
    if cfg['model_name'] in ['transformer', 'conv_lstm']:
        for split in dataset:
            if isinstance(dataset[split], dict):
                for item in dataset[split]:
                    dataset[split][item] = batchify(dataset[split][item], cfg['batch_size'][split])
            else:
                dataset[split] = batchify(dataset[split], cfg['batch_size'][split])
    return


def process_control():
    cfg['depth'] = int(cfg['control']['depth'])
    cfg['d_mode'] = [str(x) for x in cfg['control']['d_mode'].split('-')]
    cfg['d_commit'] = [float(x) for x in cfg['control']['d_commit'].split('-')]
    cfg['vqvae'] = {'hidden_size': 128, 'depth': cfg['depth'], 'num_res_block': 2, 'res_size': 32, 'embedding_size': 64,
                    'num_embedding': 512, 'vq_commit': 0.25}
    cfg['transformer'] = {'embedding_size': 16, 'num_heads': 1, 'hidden_size': 16, 'num_layers': 2,
                          'dropout': 0.2}
    cfg['conv_lstm'] = {'output_size': 16, 'num_layers': 4, 'embedding_size': 16}
    cfg['conv_lstm']['input_size'] = cfg['conv_lstm']['embedding_size']
    cfg['data_shape'] = [3, 128, 128, 128]
    if cfg['data_name'] in ['Turb']:
        if cfg['model_name'] in ['vqvae']:
            cfg['batch_size'] = {'train': 1, 'test': 1}
            cfg['num_epochs'] = 200
            cfg['optimizer_name'] = 'Adam'
            cfg['lr'] = 1e-3
            cfg['weight_decay'] = 5e-4
            cfg['scheduler_name'] = 'ReduceLROnPlateau'
        elif cfg['model_name'] in ['conv_lstm']:
            cfg['batch_size'] = {'train': 1, 'test': 1}
            cfg['num_epochs'] = 250
            cfg['optimizer_name'] = 'Adam'
            cfg['lr'] = 1e-3
            cfg['weight_decay'] = 5e-4
            cfg['scheduler_name'] = 'ReduceLROnPlateau'
            cfg['bptt'] = 1
        elif cfg['model_name'] in ['transformer']:
            cfg['batch_size'] = {'train': 1, 'test': 1}
            cfg['num_epochs'] = 250
            cfg['optimizer_name'] = 'Adam'
            cfg['lr'] = 1e-3
            cfg['weight_decay'] = 5e-4
            cfg['scheduler_name'] = 'ReduceLROnPlateau'
            cfg['bptt'] = 4
        else:
            raise ValueError('Not valid model name')
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


def make_optimizer(model):
    if cfg['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                                  weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer):
    if cfg['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs']['global'],
                                                         eta_min=cfg['min_lr'])
    elif cfg['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg['factor'],
                                                         patience=cfg['patience'], verbose=True,
                                                         threshold=cfg['threshold'], threshold_mode='rel',
                                                         min_lr=cfg['min_lr'])
    elif cfg['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg['lr'], max_lr=10 * cfg['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


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


def batchify(dataset, batch_size):
    num_batch = len(dataset) // batch_size
    dataset = dataset.narrow(0, 0, num_batch * batch_size)
    dataset = dataset.reshape(batch_size, -1, *dataset.size()[1:])
    return dataset


def Compute_1D_PDF(Signal, num_bins=int(1500)):
    """
    input= a signal 
    function= computing histogram of signal
    output= bins, frequencies
    
    """
    p, x = np.histogram((Signal.ravel() - np.mean(Signal.ravel())) / np.std(Signal.ravel()), density=True,
                        bins=num_bins)
    x = x[:-1] + (x[1] - x[0]) / 2
    p[p == 0] = np.min(p[np.nonzero(p)])
    y = np.log10(p)

    return x, y


def Compute_2D_PDF(Signal_X, Signal_Y, binwidth=12):
    """
    input= two signals 
    function= computing 2d histogram of signal
    output= bins, frequencies
    
    """
    bins_1 = np.arange(np.amin([np.amin(Signal_X), np.amin(Signal_Y)]), \
                       np.amax([np.amax(Signal_X), np.amax(Signal_Y)]) + binwidth, binwidth)

    ### histogram
    H, xedges_R, yedges_Q = np.histogram2d(Signal_X.ravel(), Signal_Y.ravel(), density=True, bins=bins_1)
    xedges_C_R = xedges_R[:-1] + (xedges_R[1] - xedges_R[0]) / 2  # convert bin edges to centers
    yedges_C_Q = yedges_Q[:-1] + (yedges_Q[1] - yedges_Q[0]) / 2  # convert bin edges to centers
    X_M, Y_M = np.meshgrid(xedges_C_R, yedges_C_Q)
    H = H.T  # Let each row list bins with common y range.

    return X_M, Y_M, H


def Compute_V_Statistics(u, v, w, Ng=128):
    """
    input: three components of velocity field
    
    output: Energy_Spectrum[Ng],
    """
    ## This part is always constant
    kk = np.fft.fftfreq(Ng, 1. / Ng)
    K = np.array(np.meshgrid(kk, kk, kk, indexing='ij'), dtype=int)
    K2 = np.sum(K * K, 0, dtype=int)
    nshell_GridP = (((2.0 * np.sqrt(K2)) + 1) // 2).astype(int)
    ##
    #####################
    Energy_GridP = np.zeros_like(u)
    Energy_k = np.zeros(u.shape[0])

    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    w_hat = np.fft.fftn(w)

    Energy_GridP = 1 / 2 * (np.real(u_hat * np.conj(u_hat)) + \
                            np.real(v_hat * np.conj(v_hat)) + \
                            np.real(w_hat * np.conj(w_hat))) / (u.shape[0]) ** 6
    for Nshell in np.unique(nshell_GridP.ravel()):
        Energy_k[Nshell] = np.sum(Energy_GridP[nshell_GridP == Nshell])
    return Energy_k


def Compute_VG_Statistics(List_VG):
    """
    input: a list consists of 9 compoents of VG tensor with shape (128,128,128)
    output: A dictionary contains Trace_A, Q, R, S_ijS_ij, R_ijR_ij, VortexStret and SijSkjSji;  all with shape (128,
    128,128)
    
    """
    # check their shape
    Ng = 128
    for item in List_VG:
        assert item.shape == (Ng, Ng, Ng), f'"""\n input.shape is {item.shape}\n""" but not {(128, 128, 128)}'

    ############# 3d
    A_3d = np.empty((Ng ** 3, 3, 3))
    A_3d[:, 0, 0] = List_VG[0].reshape(-1)
    A_3d[:, 0, 1] = List_VG[1].reshape(-1)
    A_3d[:, 0, 2] = List_VG[2].reshape(-1)
    A_3d[:, 1, 0] = List_VG[3].reshape(-1)
    A_3d[:, 1, 1] = List_VG[4].reshape(-1)
    A_3d[:, 1, 2] = List_VG[5].reshape(-1)
    A_3d[:, 2, 0] = List_VG[6].reshape(-1)
    A_3d[:, 2, 1] = List_VG[7].reshape(-1)
    A_3d[:, 2, 2] = List_VG[8].reshape(-1)

    # Trace_A
    Trace_A = np.trace(A_3d, axis1=1, axis2=2)

    # compute R and Q
    A2_3d = np.matmul(A_3d, A_3d)  # np.tensordot(A,A,axes=([4,3],[3,4]))
    A2_3d_trace = np.trace(A2_3d, axis1=1, axis2=2)
    A2_3d_trace = A2_3d_trace.reshape(Ng, Ng, Ng)
    A3_3d = np.matmul(A2_3d, A_3d)
    A3_3d_trace = np.trace(A3_3d, axis1=1, axis2=2)
    A3_3d_trace = A3_3d_trace.reshape(Ng, Ng, Ng)

    Q = (-1 / 2) * A2_3d_trace
    R = (-1 / 3) * A3_3d_trace

    # compute S_ijS_ij, R_ijR_ij
    S_3d = (1 / 2) * (A_3d + A_3d.transpose(0, 2, 1))
    Rot_3d = (1 / 2) * (A_3d - A_3d.transpose(0, 2, 1))

    SijSij_3d = np.sum(S_3d * S_3d, axis=(1, 2)).reshape(Ng, Ng, Ng)
    RijRij_3d = np.sum(Rot_3d * Rot_3d, axis=(1, 2)).reshape(Ng, Ng, Ng)

    # compute SijSkjSji, VortexStret    
    Omega_2d = np.empty((Ng ** 3, 3, 1))
    Omega_2d[:, 0, 0] = 2 * Rot_3d[:, 2, 1]
    Omega_2d[:, 1, 0] = 2 * Rot_3d[:, 0, 2]
    Omega_2d[:, 2, 0] = 2 * Rot_3d[:, 1, 0]
    VS_3d = np.matmul(S_3d, Omega_2d)

    VortexStret = np.matmul(Omega_2d.transpose(0, 2, 1), VS_3d).reshape(Ng, Ng, Ng)
    SijSkjSji = np.sum(np.matmul(S_3d, S_3d) * S_3d, axis=(1, 2)).reshape(Ng, Ng, Ng)
    Dict_VG_Stat_Outputs = {'Trace_A': Trace_A, 'Q': Q, 'R': R, 'S_ijS_ij': SijSij_3d, 'R_ijR_ij': RijRij_3d, \
                            'VS': VortexStret, 'SijSkjSji': SijSkjSji}
    return Dict_VG_Stat_Outputs


def K2_modified(Ng=128):
    kk = np.fft.fftfreq(Ng, 1. / Ng).astype(int)
    K = np.array(np.meshgrid(kk, kk, kk, indexing='ij'), dtype=int)
    K2 = np.sum(K * K, 0, dtype=int)
    return np.where(K2 == 0, 1, K2).astype(float)


K2_m = K2_modified()


def filtering_Gaussian(Phy_Sig, factor_I_L=[1e-16, 1 / 4, 1 / 2][0], \
                       K2_m=K2_m, Coef_Gauss_Filter=0.5, Ng=128, eta=1.46 / 55.8, L=1.48):
    """
    Input: Signal, filter width ([1e-16: no fliter , 1/4: inertial scales , 1/2: large scales])
    Ouput: Filtered Signal, MSE between the input and filtered input
    """

    cut_off_freq = (2 * np.pi / (factor_I_L * L))  # 1*(0.1)*(factor_I_L)/(eta)    

    Gaussina_LPF = np.exp(-Coef_Gauss_Filter * K2_m / (cut_off_freq ** 2))

    Spec_Sig = np.fft.fftn(Phy_Sig)
    filtered_Phy_Sig = np.real(np.fft.ifftn(np.multiply(Gaussina_LPF, Spec_Sig)))

    MSE_filtered = np.mean((filtered_Phy_Sig - Phy_Sig) ** 2)

    return filtered_Phy_Sig, MSE_filtered


def Filtered_Field(u, v, w):
    """
    input: 3 components of velocity field
    output: a dictionary containing intertial and large scales of input signals
    example: Filtered_Dict[name+'_LargeScales'][0] and Filtered_Dict[name+'_LargeScales'][1] \
    outputs large scale field and its MSE (compared to the original field), respectively
    
    
    """
    Filtered_Dict = {}
    for vel, name in zip([u, v, w], ['U', 'V', 'W']):
        Filtered_Dict[name + '_NoFilter'] = vel, 0.0
        Filtered_Dict[name + '_InertialScales'] = filtering_Gaussian(vel, factor_I_L=[1e-16, 1 / 4, 1 / 2][1])
        Filtered_Dict[name + '_LargeScales'] = filtering_Gaussian(vel, factor_I_L=[1e-16, 1 / 4, 1 / 2][2])
    return Filtered_Dict


def Filtered_VG(List_VG):
    """
    input: a list consists of 9 components of velocity gradient tensor
    output: a dictionary containing intertial and large scales of input signals
    example: Filtered_Dict[name+'_LargeScales'][0] and Filtered_Dict[name+'_LargeScales'][1] \
    outputs large scale field and its MSE (compared to the original field), respectively
    
    """
    str_list_var = ['dUdx_Phy', 'dUdy_Phy', 'dUdz_Phy', 'dVdx_Phy', 'dVdy_Phy', 'dVdz_Phy', 'dWdx_Phy', 'dWdy_Phy',
                    'dWdz_Phy']

    Filtered_Dict = {}
    for vel_g, name in zip(List_VG, str_list_var):
        Filtered_Dict[name + '_NoFilter'] = vel_g, 0.0
        Filtered_Dict[name + '_InertialScales'] = filtering_Gaussian(vel_g, factor_I_L=[1e-16, 1 / 4, 1 / 2][1])
        Filtered_Dict[name + '_LargeScales'] = filtering_Gaussian(vel_g, factor_I_L=[1e-16, 1 / 4, 1 / 2][2])

    # compute VG statistics 
    ## for NoFilter
    Dict_VG_Stat_Outputs = {}
    Dict_VG_Stat_Outputs = Compute_VG_Statistics([Filtered_Dict[item + '_NoFilter'][0] for item in str_list_var])
    for item in Dict_VG_Stat_Outputs:
        Filtered_Dict[item + '_NoFilter'] = Dict_VG_Stat_Outputs[item]
        ### for InertialScales
    Dict_VG_Stat_Outputs = {}
    Dict_VG_Stat_Outputs = Compute_VG_Statistics([Filtered_Dict[item + '_InertialScales'][0] for item in str_list_var])
    for item in Dict_VG_Stat_Outputs:
        Filtered_Dict[item + '_InertialScales'] = Dict_VG_Stat_Outputs[item]
        #### for LargeScales
    Dict_VG_Stat_Outputs = {}
    Dict_VG_Stat_Outputs = Compute_VG_Statistics([Filtered_Dict[item + '_LargeScales'][0] for item in str_list_var])
    for item in Dict_VG_Stat_Outputs:
        Filtered_Dict[item + '_LargeScales'] = Dict_VG_Stat_Outputs[item]

        # print(Filtered_Dict.keys())

    return Filtered_Dict


def vis(input, output, path, i_d_min=5, fontsize=10, num_bins=1500):
    from matplotlib.pyplot import contour, contourf
    import scipy.stats as stats

    input_uvw = input['uvw'].cpu().numpy()
    input_duvw = input['duvw'].cpu().numpy()
    output_uvw = output['uvw'].cpu().numpy()
    output_duvw = output['duvw'].cpu().numpy()

    j_d_min, j_d_max = 0, 128
    k_d_min, k_d_max = 0, 128
    label = ['U', 'V', 'W']
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    xx = np.linspace(-5, 5, 1000)
    yy = np.log10(stats.norm.pdf(xx, 0, 1))
    for i in range(3):
        plt.colorbar(ax[i][0].imshow(input_uvw[0, i, i_d_min:(i_d_min + 1), j_d_min:j_d_max,
                                     k_d_min:k_d_max].squeeze()), ax=ax[i][0], fraction=0.046, pad=0.04)
        plt.colorbar(ax[i][1].imshow(output_uvw[0, i, i_d_min:(i_d_min + 1), j_d_min:j_d_max,
                                     k_d_min:k_d_max].squeeze()), ax=ax[i][1], fraction=0.046, pad=0.04)
        ax[i][0].set_title('Original {}'.format(label[i]), fontsize=fontsize)
        ax[i][1].set_title('Reconstructed {}'.format(label[i]), fontsize=fontsize)
        x, y = Compute_1D_PDF(input_uvw[0, i, :, :, :], num_bins=num_bins)

        ax[i][2].plot(x, y, 'b', lw=2, label='Original {}'.format(label[i]))
        x, y = Compute_1D_PDF(output_uvw[0, i, :, :, :], num_bins=num_bins)

        ax[i][2].plot(x, y, 'g', lw=2, label='Reconstructed {}'.format(label[i]))
        ax[i][2].set_xlim(-10, 10)
        ax[i][2].set_ylim(-5, 0)
        ax[i][2].set_xlabel('Normalized {}'.format(label[i]), fontsize=fontsize)
        ax[i][2].set_ylabel('log10(pdf)', fontsize=fontsize)
        ax[i][2].set_title('MSE = {:.4f}'.format(np.mean((output_uvw[:, i, :, :, :] - input_uvw[:, i, :, :, :]) ** 2)),
                           fontsize=fontsize)
        ax[i][2].grid(True)

        ax[i][2].plot(xx, yy, 'r--', label="Gaussian")
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
            x, y = Compute_1D_PDF(input_duvw[:, i, j, :, :, :], num_bins=num_bins)

            ax[i][j].plot(x, y, 'g', lw=2, label='Original {}'.format(label[i][j]))
            x, y = Compute_1D_PDF(output_duvw[:, i, j, :, :, :], num_bins=num_bins)

            ax[i][j].plot(x, y, 'b', lw=2, label='Reconstructed {}'.format(label[i][j]))
            ax[i][j].set_title('MSE = {:.4f}'.format(np.mean((output_duvw[:, i, j, :, :, :] -
                                                              input_duvw[:, i, j, :, :, :]) ** 2)), fontsize=fontsize)
            ax[i][j].set_xlim(-10, 10)
            ax[i][j].set_ylim(-5, 0)
            ax[i][j].set_xlabel('Normalized {}'.format(label[i][j]), fontsize=fontsize)
            ax[i][j].set_ylabel('log10(PDF)', fontsize=fontsize)
            ax[i][j].grid(True)

            ax[i][j].plot(xx, yy, 'r--', label="Gaussian")
            ax[i][j].legend(fontsize=fontsize)
    plt.tight_layout()
    makedir_exist_ok(path)
    fig.savefig('{}/vg_{}.{}'.format(path, cfg['model_tag'], cfg['fig_format']), dpi=300, bbox_inches='tight',
                fontsize=fontsize)
    plt.close()

    Ng = 128
    # Velocity field Statistics

    Energy_k_Original = Compute_V_Statistics(input_uvw[0, 0, :, :, :], input_uvw[0, 1, :, :, :],
                                             input_uvw[0, 2, :, :, :])
    Energy_k_Reconstructed = Compute_V_Statistics(output_uvw[0, 0, :, :, :], output_uvw[0, 1, :, :, :],
                                                  output_uvw[0, 2, :, :, :])

    # Plot Energy Spectrum
    O_color = 'blue'
    R_color = 'red'
    title = ['Original', 'Reconstructed']
    fontsize_text = 18
    fig = plt.figure(figsize=(8, 6))
    fontsize_label = 20
    xx = np.arange(Ng)
    plt.plot(xx, Energy_k_Original, color=O_color, label=title[0])
    plt.plot(xx, Energy_k_Reconstructed, color=R_color, linestyle='--', label=title[1])
    plt.plot(xx[1:], (xx[1:] ** (-5 / 3)), color='k', label='$K^{(-5/3)}$')
    plt.legend(fontsize=fontsize_text)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([1e-6, 10])
    plt.xlabel(r"$k$", fontsize=fontsize_label)
    plt.ylabel(r"$E(k)$", fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_label)
    plt.yticks(fontsize=fontsize_label)
    plt.grid()

    plt.tight_layout()
    makedir_exist_ok(path)
    fig.savefig('{}/EnergySpectrum_{}.{}'.format(path, cfg['model_tag'], cfg['fig_format']), dpi=300,
                bbox_inches='tight',
                fontsize=fontsize)
    plt.close()

    # Velocity Gradient Statistics
    VG_Stat_Original = VG_Stat_Recons = {}
    VG_Stat_Original = Compute_VG_Statistics(
        [input_duvw[0, 0, 0, :, :, :], input_duvw[0, 0, 1, :, :, :], input_duvw[0, 0, 2, :, :, :], \
         input_duvw[0, 1, 0, :, :, :], input_duvw[0, 1, 1, :, :, :], input_duvw[0, 1, 2, :, :, :], \
         input_duvw[0, 2, 0, :, :, :], input_duvw[0, 2, 1, :, :, :], input_duvw[0, 2, 2, :, :, :]])
    VG_Stat_Recons = Compute_VG_Statistics(
        [output_duvw[0, 0, 0, :, :, :], output_duvw[0, 0, 1, :, :, :], output_duvw[0, 0, 2, :, :, :], \
         output_duvw[0, 1, 0, :, :, :], output_duvw[0, 1, 1, :, :, :], output_duvw[0, 1, 2, :, :, :], \
         output_duvw[0, 2, 0, :, :, :], output_duvw[0, 2, 1, :, :, :], output_duvw[0, 2, 2, :, :, :]])
    # plot R-Q
    lev = np.array([1e-1 * 0.001, 1e-2 * 0.001, 1e-3 * 0.001, 1e-4 * 0.001])[::-1]
    extend = 6
    O_color = 'blue'
    R_color = 'red'
    camp = 'viridis'  # 'hot'

    fig = plt.figure(figsize=(12, 8))

    X_M, Y_M, H = Compute_2D_PDF(VG_Stat_Original['R'], VG_Stat_Original['Q'])
    SijSij_mean_t = np.mean(VG_Stat_Original['S_ijS_ij'])

    contours = contour(X_M / SijSij_mean_t ** (3 / 2), Y_M / SijSij_mean_t, H, levels=lev, \
                       origin='lower', colors=4 * (O_color,), linewidths=1, linestyles='solid')
    plt.clabel(contours, inline=True, fmt='%.9f', fontsize=14)

    X_M, Y_M, H = Compute_2D_PDF(VG_Stat_Recons['R'], VG_Stat_Recons['Q'])
    SijSij_mean_t = np.mean(VG_Stat_Recons['S_ijS_ij'])

    contours = contour(X_M / SijSij_mean_t ** (3 / 2), Y_M / SijSij_mean_t, H, levels=lev, \
                       origin='lower', colors=4 * (R_color,), linewidths=3, linestyles='--')
    plt.clabel(contours, inline=True, fmt='%.9f', fontsize=14)

    Rx = np.arange(-10, 10, 0.1)
    plt.plot(Rx, -((27 / 4) * Rx ** 2) ** (1 / 3), 'k-', label='$Q=-(27R^2/4)^{1/3}$')
    plt.legend(fontsize=15)

    plt.xlabel(r"$R/(S_{ij}S_{ij})^{(3/2)}}$", fontsize=fontsize_label)
    plt.ylabel(r"$Q/S_{ij}S_{ij}$", fontsize=fontsize_label)
    plt.xlim([-extend, extend])
    plt.ylim([-extend, extend])
    plt.grid()

    plt.tight_layout()
    makedir_exist_ok(path)
    fig.savefig('{}/RQ_{}.{}'.format(path, cfg['model_tag'], cfg['fig_format']), dpi=300, bbox_inches='tight',
                fontsize=fontsize)
    plt.close()

    # Plot other VG statistics
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    x_st = 0.1
    y_st = 1.75
    step = 0.3
    fontsize_text = 18
    title = ['Original', 'Reconstructed']
    i = 0
    for dic in [VG_Stat_Original, VG_Stat_Recons]:
        axes[i].scatter([0, 1], [2, 0], color='w')
        axes[i].text(x_st, y_st, r'$|A_{ii}| = %.4e$' % np.mean(dic['Trace_A']), fontsize=fontsize_text)
        axes[i].text(x_st, y_st - 1 * step, r'$|S_{ij}S_{ij}| = %.4f$' % np.mean(dic['S_ijS_ij']),
                     fontsize=fontsize_text)
        axes[i].text(x_st, y_st - 2 * step, r'$|R_{ij}R_{ij}| = %.4f$' % np.mean(dic['R_ijR_ij']),
                     fontsize=fontsize_text)
        axes[i].text(x_st, y_st - 3 * step,
                     r'$(-3/4)*|S_{ij}\omega_i\omega_j| = %.4f$' % ((-3 / 4) * np.mean(dic['VS'])),
                     fontsize=fontsize_text)
        axes[i].text(x_st, y_st - 4 * step, r'$|S_{ij}S_{kj}S_{ji}| = %.4f$' % np.mean(dic['SijSkjSji']),
                     fontsize=fontsize_text)

        axes[i].axes.xaxis.set_visible(False)
        axes[i].axes.yaxis.set_visible(False)
        axes[i].set_title("%s" % title[i], fontsize=fontsize_text)

        i += 1

    plt.tight_layout()
    makedir_exist_ok(path)
    fig.savefig('{}/VG_SummaryStatistics_{}.{}'.format(path, cfg['model_tag'], cfg['fig_format']), dpi=300,
                bbox_inches='tight',
                fontsize=fontsize)
    plt.close()

    # Filtering Statistics
    Original_Filtered_Field = Filtered_Field(input_uvw[0, 0, :, :, :], input_uvw[0, 1, :, :, :],
                                             input_uvw[0, 2, :, :, :])
    Reconstructed_Filtered_Field = Filtered_Field(output_uvw[0, 0, :, :, :], output_uvw[0, 1, :, :, :],
                                                  output_uvw[0, 2, :, :, :])

    ## Plot U,V,W filtered
    j_d_min, j_d_max = 0, 128
    k_d_min, k_d_max = 0, 128
    i_d_min = np.random.randint(0, Ng)  # let's keep it random and not stick to a specific cross-section
    xx = np.linspace(-5, 5, 1000)
    yy = np.log10(stats.norm.pdf(xx, 0, 1))

    for Vel_comp in ['U', 'V', 'W']:
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
        for i, name in zip(range(3), ['_NoFilter', '_InertialScales', '_LargeScales']):
            plt.colorbar(ax[0][i].imshow(
                Original_Filtered_Field[Vel_comp + name][0][i_d_min:(i_d_min + 1), j_d_min:j_d_max,
                k_d_min:k_d_max].squeeze()), ax=ax[0][i], fraction=0.046, pad=0.04)
            plt.colorbar(ax[1][i].imshow(
                Reconstructed_Filtered_Field[Vel_comp + name][0][i_d_min:(i_d_min + 1), j_d_min:j_d_max,
                k_d_min:k_d_max].squeeze()), ax=ax[1][i], fraction=0.046, pad=0.04)

            ax[0][i].set_title('{} Original{}'.format(Vel_comp, name) \
                                   if i == 0 else \
                                   '{} Original{} , MSE = {:.4f}'. \
                               format(Vel_comp, name, Original_Filtered_Field[Vel_comp + name][1]), fontsize=fontsize)
            ax[1][i].set_title('{} Reconstructed{}'.format(Vel_comp, name) \
                                   if i == 0 else \
                                   '{} Reconstructed{} , MSE = {:.4f}'. \
                               format(Vel_comp, name, Reconstructed_Filtered_Field[Vel_comp + name][1]),
                               fontsize=fontsize)

            x, y = Compute_1D_PDF(Original_Filtered_Field[Vel_comp + name][0], num_bins=num_bins)

            ax[2][i].plot(x, y, 'b', lw=2, label='{} Original{}'.format(Vel_comp, name))
            x, y = Compute_1D_PDF(Reconstructed_Filtered_Field[Vel_comp + name][0], num_bins=num_bins)

            ax[2][i].plot(x, y, 'g', lw=2, label='{} Reconstructed{}'.format(Vel_comp, name))
            ax[2][i].set_xlim(-10, 10)
            ax[2][i].set_ylim(-5, 0)
            ax[2][i].set_xlabel('Normalized {}'.format(Vel_comp), fontsize=fontsize)
            ax[2][i].set_ylabel('log10(pdf)', fontsize=fontsize)
            ax[2][i].set_title('MSE = {:.4f}'.format(np.mean(
                (Reconstructed_Filtered_Field[Vel_comp + name][0] - Original_Filtered_Field[Vel_comp + name][0]) ** 2)),
                fontsize=fontsize)
            ax[2][i].grid(True)

            ax[2][i].plot(xx, yy, 'r--', label="Gaussian")
            ax[2][i].legend(fontsize=fontsize)
        plt.tight_layout()
        fig.savefig('{}/{}_Summary_{}.{}'.format(path, Vel_comp, cfg['model_tag'], cfg['fig_format']), dpi=300,
                    bbox_inches='tight',
                    fontsize=fontsize)

        plt.close()

    ## plot R-Q filtered
    Original_Filtered_VG_Field = Filtered_VG(
        [input_duvw[0, 0, 0, :, :, :], input_duvw[0, 0, 1, :, :, :], input_duvw[0, 0, 2, :, :, :], \
         input_duvw[0, 1, 0, :, :, :], input_duvw[0, 1, 1, :, :, :], input_duvw[0, 1, 2, :, :, :], \
         input_duvw[0, 2, 0, :, :, :], input_duvw[0, 2, 1, :, :, :], input_duvw[0, 2, 2, :, :, :]])
    Reconstructed_Filtered_VG_Field = Filtered_VG(
        [output_duvw[0, 0, 0, :, :, :], output_duvw[0, 0, 1, :, :, :], output_duvw[0, 0, 2, :, :, :], \
         output_duvw[0, 1, 0, :, :, :], output_duvw[0, 1, 1, :, :, :], output_duvw[0, 1, 2, :, :, :], \
         output_duvw[0, 2, 0, :, :, :], output_duvw[0, 2, 1, :, :, :], output_duvw[0, 2, 2, :, :, :]])

    ### 
    for Scale in ['_NoFilter', '_InertialScales', '_LargeScales']:
        fig = plt.figure(figsize=(12, 8))
        X_M, Y_M, H = Compute_2D_PDF(Original_Filtered_VG_Field['R' + Scale], Original_Filtered_VG_Field['Q' + Scale])
        SijSij_mean_t = np.mean(Original_Filtered_VG_Field['S_ijS_ij' + Scale])

        contours = contour(X_M / SijSij_mean_t ** (3 / 2), Y_M / SijSij_mean_t, H, levels=lev, \
                           origin='lower', colors=4 * (O_color,), linewidths=1, linestyles='solid')

        X_M, Y_M, H = Compute_2D_PDF(Reconstructed_Filtered_VG_Field['R' + Scale],
                                     Reconstructed_Filtered_VG_Field['Q' + Scale])
        SijSij_mean_t = np.mean(Reconstructed_Filtered_VG_Field['S_ijS_ij' + Scale])

        contours = contour(X_M / SijSij_mean_t ** (3 / 2), Y_M / SijSij_mean_t, H, levels=lev, \
                           origin='lower', colors=4 * (R_color,), linewidths=3, linestyles='--')

        Rx = np.arange(-10, 10, 0.1)
        plt.plot(Rx, -((27 / 4) * Rx ** 2) ** (1 / 3), 'k-', label='$Q=-(27R^2/4)^{1/3}$')
        plt.legend(fontsize=15)

        plt.xlabel(r"$R/(S_{ij}S_{ij})^{(3/2)}}$", fontsize=fontsize_label)
        plt.ylabel(r"$Q/S_{ij}S_{ij}$", fontsize=fontsize_label)
        plt.xlim([-extend, extend])
        plt.ylim([-extend, extend])
        plt.grid()
        plt.title("R-Q" + Scale)
        plt.tight_layout()
        makedir_exist_ok(path)
        fig.savefig('{}/RQ{}_{}.{}'.format(path, Scale, cfg['model_tag'], cfg['fig_format']), dpi=300,
                    bbox_inches='tight',
                    fontsize=fontsize)

        plt.close()

    #### Plot other Filtered VG statistics
    x_st = 0.1
    y_st = 1.75
    step = 0.3
    fontsize_text = 18
    title = ['Original', 'Reconstructed']
    for Scale in ['_NoFilter', '_InertialScales', '_LargeScales']:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        i = 0
        for dic in [Original_Filtered_VG_Field, Reconstructed_Filtered_VG_Field]:
            axes[i].scatter([0, 1], [2, 0], color='w')
            axes[i].text(x_st, y_st, r'$|A_{ii}| = %.4e$' % np.mean(dic['Trace_A' + Scale]), fontsize=fontsize_text)
            axes[i].text(x_st, y_st - 1 * step, r'$|S_{ij}S_{ij}| = %.4f$' % np.mean(dic['S_ijS_ij' + Scale]),
                         fontsize=fontsize_text)
            axes[i].text(x_st, y_st - 2 * step, r'$|R_{ij}R_{ij}| = %.4f$' % np.mean(dic['R_ijR_ij' + Scale]),
                         fontsize=fontsize_text)
            axes[i].text(x_st, y_st - 3 * step,
                         r'$(-3/4)*|S_{ij}\omega_i\omega_j| = %.4f$' % ((-3 / 4) * np.mean(dic['VS' + Scale])),
                         fontsize=fontsize_text)
            axes[i].text(x_st, y_st - 4 * step, r'$|S_{ij}S_{kj}S_{ji}| = %.4f$' % np.mean(dic['SijSkjSji' + Scale]),
                         fontsize=fontsize_text)

            axes[i].axes.xaxis.set_visible(False)
            axes[i].axes.yaxis.set_visible(False)
            axes[i].set_title("%s" % (title[i] + Scale), fontsize=fontsize_text)

            i += 1

        plt.tight_layout()
        makedir_exist_ok(path)
        fig.savefig('{}/VG{}_SummaryStatistics_{}.{}'.format(path, Scale, cfg['model_tag'], cfg['fig_format']), dpi=300,
                    bbox_inches='tight',
                    fontsize=fontsize)
        plt.close()

    label = [['dUdx', 'dUdy', 'dUdz'], ['dVdx', 'dVdy', 'dVdz'], ['dWdx', 'dWdy', 'dWdz']]
    for Scale in ['_NoFilter', '_InertialScales', '_LargeScales']:
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 25))
        fontsize = 15
        for i in range(3):
            for j in range(3):
                x, y = Compute_1D_PDF(Original_Filtered_VG_Field[label[i][j] + '_Phy' + Scale][0], num_bins=num_bins)

                ax[i][j].plot(x, y, 'g', lw=2, label='Original {}'.format(label[i][j]))
                x, y = Compute_1D_PDF(Reconstructed_Filtered_VG_Field[label[i][j] + '_Phy' + Scale][0],
                                      num_bins=num_bins)

                ax[i][j].plot(x, y, 'b', lw=2, label='Reconstructed {}'.format(label[i][j]))
                ax[i][j].set_title('{}{} MSE = {:.4f}'.format(label[i][j], Scale, \
                                                              np.mean((Original_Filtered_VG_Field[
                                                                           label[i][j] + '_Phy' + Scale][0] -
                                                                       Reconstructed_Filtered_VG_Field[
                                                                           label[i][j] + '_Phy' + Scale][0]) ** 2)),
                                   fontsize=fontsize)
                ax[i][j].set_xlim(-10, 10)
                ax[i][j].set_ylim(-5, 0)
                ax[i][j].set_xlabel('Normalized {}'.format(label[i][j]), fontsize=fontsize)
                ax[i][j].set_ylabel('log10(PDF)', fontsize=fontsize)
                ax[i][j].grid(True)

                ax[i][j].plot(xx, yy, 'r--', label="Gaussian")
                ax[i][j].legend(fontsize=fontsize)
        plt.tight_layout()

        makedir_exist_ok(path)
        fig.savefig('{}/VG{}_PDF_{}.{}'.format(path, Scale, cfg['model_tag'], cfg['fig_format']), dpi=300,
                    bbox_inches='tight',
                    fontsize=fontsize)
        plt.close()

    return