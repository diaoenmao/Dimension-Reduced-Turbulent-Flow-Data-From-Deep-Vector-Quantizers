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
        cfg['hidden_size'] = [64, 128, 256]
        cfg['quantizer_embedding_size'] = 128
        cfg['num_embedding'] = 512
        cfg['vq_commit'] = 0.25
    return


def make_stats(dataset):
    if os.path.exists('./data/stats/{}.pt'.format(dataset.data_name)):
        stats = load('./data/stats/{}.pt'.format(dataset.data_name))
    elif dataset is not None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
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


# def vis(signal, recon_signal, path, i_d_min=5, fontsize=10, num_bins=1500):
#     import scipy.stats as stats
    
#     print("signal.shape = ")#, signal.shape())
#     print(signal.shape)
    
#     fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
#     j_d_min, j_d_max = 0, 128
#     k_d_min, k_d_max = 0, 128
#     label = ['U', 'V', 'W']
#     for i in range(3):
#         plt.colorbar(ax[i][0].imshow(signal[0, i, i_d_min:(i_d_min + 1), j_d_min:j_d_max,
#                                      k_d_min:k_d_max].squeeze()), ax=ax[i][0], fraction=0.046, pad=0.04)
#         plt.colorbar(ax[i][1].imshow(recon_signal[0, i, i_d_min:(i_d_min + 1), j_d_min:j_d_max,
#                                      k_d_min:k_d_max].squeeze()), ax=ax[i][1], fraction=0.046, pad=0.04)
#         ax[i][0].set_title('Original {}'.format(label[i]), fontsize=fontsize)
#         ax[i][1].set_title('Reconstructed {}'.format(label[i]), fontsize=fontsize)

#         p, x = np.histogram((signal[0, i,:,:,:].ravel() - np.mean(signal[0, i,:,:,:].ravel())) / np.std(signal[0, i,:,:,:].ravel()), density=True,
#                             bins=num_bins)
#         x = x[:-1] + (x[1] - x[0]) / 2
#         p[p == 0] = np.min(p[np.nonzero(p)])
#         y = np.log10(p)
#         ax[i][2].plot(x, y, 'b', lw=2, label='Original {}'.format(label[i]))
#         p, x = np.histogram((recon_signal[0, i,:,:,:].ravel() - np.mean(recon_signal[0, i,:,:,:].ravel())) / np.std(recon_signal[0, i,:,:,:].ravel()),
#                             density=True, bins=num_bins)
#         x = x[:-1] + (x[1] - x[0]) / 2
#         p[p == 0] = np.min(p[np.nonzero(p)])
#         y = np.log10(p)
#         ax[i][2].plot(x, y, 'g', lw=2, label='Reconstructed {}'.format(label[i]))
#         ax[i][2].set_xlim(-6, 6)
#         ax[i][2].set_ylim(-5, 0)
#         ax[i][2].set_xlabel('Normalized Signal', fontsize=fontsize)
#         ax[i][2].set_ylabel('log10(pdf)', fontsize=fontsize)
#         ax[i][2].grid(True)
#         xx = np.linspace(-5, 5, 1000)
#         ax[i][2].plot(xx, np.log10(stats.norm.pdf(xx, 0, 1)), 'r--', label="Gaussian")
#         ax[i][2].legend(fontsize=fontsize)
#     plt.tight_layout()
#     dir = os.path.dirname(path)
#     makedir_exist_ok(dir)
#     fig.savefig(path, dpi=300, bbox_inches='tight', fontsize=fontsize)
#     plt.close()
#     return
def save_obj(obj, dir_path,obj_name):
    import pickle
    with open(dir_path+'/'+ obj_name+ '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def FFT_derivative(Phy_Vel,direction='x',Fast=True):
    kk=np.fft.fftfreq (Phy_Vel.shape[0] , 1./ Phy_Vel.shape[0])
    pre=np.fft.fftn(Phy_Vel)
    post=np.zeros_like(pre)
    output=np.zeros_like(Phy_Vel)
    
    if Fast==False:
        
        if direction=='x':
            for i in range(Phy_Vel.shape[0]):
                for j in range(Phy_Vel.shape[1]):       
                    for k in range(Phy_Vel.shape[2]):           
                        post[k,j,i] = pre[k,j,i]*1.0j*kk[i] # no matter if you put kx or ky or kz
        if direction=='y':
            for i in range(Phy_Vel.shape[0]):
                for j in range(Phy_Vel.shape[1]):       
                    for k in range(Phy_Vel.shape[2]):           
                        post[k,j,i] = pre[k,j,i]*1.0j*kk[j] # no matter if you put kx or ky or kz
        if direction=='z':
            for i in range(Phy_Vel.shape[0]):
                for j in range(Phy_Vel.shape[1]):       
                    for k in range(Phy_Vel.shape[2]):           
                        post[k,j,i] = pre[k,j,i]*1.0j*kk[k] # no matter if you put kx or ky or kz
        output=np.real(np.fft.ifftn(post))
        return output
    
    else:
        
        kxmesh,kymesh,kzmesh=np.meshgrid(kk,kk,kk,indexing='ij');
    
        if direction=='x':
            output=np.real(np.fft.ifftn(1.0j *np.multiply(pre,kzmesh)))       
        elif direction=='y':
            output=np.real(np.fft.ifftn(1.0j *np.multiply(pre,kymesh)))  
        else : #direction=='z'
            output=np.real(np.fft.ifftn(1.0j *np.multiply(pre,kxmesh)))  
        return output
        
def Q_R_Calculator_slow(A_2d):
    A2=np.matmul(A_2d,A_2d)
    A3=np.matmul(A_2d,A2)
    

    
    Q=(-1/2)*np.trace(A2)
    R=(-1/3)*np.trace(A3)
    
    S=(1/2)*(A_2d+A_2d.T)
    SijSij=np.sum(S*S) # rotation
    
    Rot=(1/2)*(A_2d-A_2d.T)
    
    RijRij=np.sum(Rot*Rot)
                     
    return Q,R,SijSij,RijRij


def Q_R_Calculator(A11,A12,A13,A21,A22,A23,A31,A32,A33,Fast=True):
    
    if Fast==False:
        ng=A11.shape[0]
        R_loop=np.zeros_like(A11)
        Q_loop=np.zeros_like(A11)
        SijSij_loop=np.zeros_like(A11)
        RijRij_loop=np.zeros_like(A11)
        
        for i in range(0,ng):
            for j in range(0,ng):
                for k in range(0,ng):
                    A_2d_t=np.array([
                        [A11[i,j,k],A12[i,j,k],A13[i,j,k]],
                        [A21[i,j,k],A22[i,j,k],A23[i,j,k]],
                        [A31[i,j,k],A32[i,j,k],A33[i,j,k]]
                    ])
                    Q_loop[i,j,k],R_loop[i,j,k],SijSij_loop[i,j,k],RijRij_loop[i,j,k]=Q_R_Calculator_slow(A_2d_t)
        
        return Q_loop,R_loop,SijSij_loop,RijRij_loop
    
    else:
        
        # initialize the outputs
        trace_A2=np.zeros_like(A11)
        trace_A3=np.zeros_like(A11)
        S_ijS_ij=np.zeros_like(A11)
        R_ijR_ij=np.zeros_like(A11)
        
        # compute trace of A2=A^2=np.matmul(A,A)
        trace_A2= A11**2 + A22**2 + A33**2 + 2*(A12*A21 + A13*A31 +A23*A32)

        # compute trace of A3=A^3=np.matmul(A,A2)
        trace_A3= A11*(A11**2 + A12*A21 + A13*A31) +\
            A22*(A22**2 + A12*A21 + A23*A32) +\
                A33*(A33**2 + A13*A31 +A23*A32)+\
                    A21*(A11*A12 + A12*A22 +A13*A32) +\
                        A31*(A11*A13 + A12*A23 +A13*A33) +\
                            A12*(A11*A21 +A21*A22 + A23*A31) +\
                                A32*(A13*A21+ A22*A23 + A23*A33) +\
                                    A13*(A11*A31+ A21*A32 + A31*A33) +\
                                        A23*(A12*A31+ A22*A32 + A32*A33)

        S_ijS_ij= ((1/2)*(A11+A11))**2+((1/2)*(A22+A22))**2+((1/2)*(A33+A33))**2+\
        2*((1/2)*(A12+A21))**2+2*((1/2)*(A13+A31))**2+2*((1/2)*(A23+A32))**2 

        R_ijR_ij= 2*((1/2)*(A12-A21))**2+2*((1/2)*(A13-A31))**2+2*((1/2)*(A23-A32))**2 


        return (-1/2)*trace_A2,(-1/3)*trace_A3,S_ijS_ij,R_ijR_ij  #Q,R,,S_ijS_ij

def plot_PDF_VelocityGrad_DL_Model_DNS(A11_DNS,A12_DNS,A13_DNS,A21_DNS,A22_DNS,A23_DNS,A31_DNS,A32_DNS,A33_DNS,\
                                       A11_Model,A12_Model,A13_Model,A21_Model,A22_Model,A23_Model,A31_Model,A32_Model,A33_Model,\
                         str_list_var=['dUdx_Phy','dUdy_Phy','dUdz_Phy','dVdx_Phy','dVdy_Phy','dVdz_Phy','dWdx_Phy','dWdy_Phy','dWdz_Phy'],\
                         num_bins = int(1000//2),dir_path='./'):
    import scipy.stats as stats
    def replaceZeroes(data):
        min_nonzero = np.min(data[np.nonzero(data)])
        data[data == 0] = min_nonzero
        return data
    
    list_var_Model=[A11_Model,A12_Model,A13_Model,A21_Model,A22_Model,A23_Model,A31_Model,A32_Model,A33_Model]
    list_var_DNS=[A11_DNS,A12_DNS,A13_DNS,A21_DNS,A22_DNS,A23_DNS,A31_DNS,A32_DNS,A33_DNS]

    fig,axes=plt.subplots(nrows=3, ncols=3,figsize=(20,25))     
    
    fontsize=15
    for i,var_Model,var_DNS,var_name in zip(range(9),list_var_Model,list_var_DNS,str_list_var):       
        
        # Model
        p, x = np.histogram((var_Model.ravel()-np.mean(var_Model.ravel()))/np.std(var_Model.ravel()), density=True,bins=num_bins) 
        x = x[:-1] + (x[1] - x[0])/2
        y=np.log10(replaceZeroes(p))
        
        axes[i//3][i%3].plot(x, y,'b',lw=2,label=var_name+'_Model')
    
        # DNS
        p, x = np.histogram((var_DNS.ravel()-np.mean(var_DNS.ravel()))/np.std(var_DNS.ravel()), density=True,bins=num_bins) 
        x = x[:-1] + (x[1] - x[0])/2
        y=np.log10(replaceZeroes(p))
        
        axes[i//3][i%3].plot(x, y,'g',lw=2,label=var_name+'_DNS')
        
        axes[i//3][i%3].set_title("MSE = %.4f" % np.mean((var_DNS-var_Model)**2) , fontsize=fontsize)
        
        axes[i//3][i%3].set_xlim(-10, 10)
        axes[i//3][i%3].set_ylim(-5, 0)
        
        axes[i//3][i%3].set_xlabel("Normalized "+var_name,fontsize=fontsize)
        axes[i//3][i%3].set_ylabel('log10(PDF)',fontsize=fontsize)
        
        axes[i//3][i%3].grid(True)
        
        xx =np.linspace(-5, 5, 1000)
        axes[i//3][i%3].plot(xx, np.log10(stats.norm.pdf(xx, 0, 1)),'r--',label="Guassian")
        axes[i//3][i%3].legend(fontsize=fontsize)
        
    if dir_path!='./':
        plt.tight_layout()
        dir = os.path.dirname(dir_path)
        makedir_exist_ok(dir)
        fig.savefig(dir_path, dpi=300, bbox_inches='tight', fontsize=fontsize)
        plt.close()
    
    
    
def vis(signal, recon_signal, path, i_d_min=5, fontsize=10, num_bins=1500):
    import scipy.stats as stats
    
    print("signal.shape = ")#, signal.shape())
    print(signal.shape)
    
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

        p, x = np.histogram((signal[0, i,:,:,:].ravel() - np.mean(signal[0, i,:,:,:].ravel())) / np.std(signal[0, i,:,:,:].ravel()), density=True,
                            bins=num_bins)
        x = x[:-1] + (x[1] - x[0]) / 2
        p[p == 0] = np.min(p[np.nonzero(p)])
        y = np.log10(p)
        ax[i][2].plot(x, y, 'b', lw=2, label='Original {}'.format(label[i]))
        p, x = np.histogram((recon_signal[0, i,:,:,:].ravel() - np.mean(recon_signal[0, i,:,:,:].ravel())) / np.std(recon_signal[0, i,:,:,:].ravel()),
                            density=True, bins=num_bins)
        x = x[:-1] + (x[1] - x[0]) / 2
        p[p == 0] = np.min(p[np.nonzero(p)])
        y = np.log10(p)
        ax[i][2].plot(x, y, 'g', lw=2, label='Reconstructed {}'.format(label[i]))
        ax[i][2].set_xlim(-6, 6)
        ax[i][2].set_ylim(-5, 0)
        ax[i][2].set_xlabel('Normalized Signal', fontsize=fontsize)
        ax[i][2].set_ylabel('log10(pdf)', fontsize=fontsize)
        ax[i][2].set_title("MSE = %.4f" % np.mean((recon_signal[0, i,:,:,:]-signal[0, i,:,:,:])**2) , fontsize=fontsize)
        ax[i][2].grid(True)
        xx = np.linspace(-5, 5, 1000)
        ax[i][2].plot(xx, np.log10(stats.norm.pdf(xx, 0, 1)), 'r--', label="Gaussian")
        ax[i][2].legend(fontsize=fontsize)
    plt.tight_layout()
    dir = os.path.dirname(path)
    makedir_exist_ok(dir)
    fig.savefig(path, dpi=300, bbox_inches='tight', fontsize=fontsize)
    plt.close()
    
    #####
    # storing the velocity components geerated by model
    DL_model_outputs_DNS={}
    for i,name in enumerate(label):
        DL_model_outputs_DNS[name+'_'+'Model']=recon_signal[0, i,:,:,:]
        DL_model_outputs_DNS[name+'_'+'DNS']=signal[0, i,:,:,:]
    
    # compute the velocity gradients
    str_list_var=['A'+str(i)+str(j) for i in range(1,3+1) for j in range(1,3+1)]
    directions=3*['x','y','z']
    list_var_Vel_name=[bb for a in [3*[item] for item in ['U', 'V', 'W']] for bb in a ]
    for var_name,deriv_dir,var_vel_name in zip(str_list_var,directions,list_var_Vel_name):
        # compute derivatives of Model outputs
        var_vel=DL_model_outputs_DNS[var_vel_name+'_'+'Model']
        der=FFT_derivative(var_vel,direction=deriv_dir,Fast=True)
        DL_model_outputs_DNS[var_name + '_' + 'Model']=der
        # compute derivatives of DNS data
        var_vel=DL_model_outputs_DNS[var_vel_name+'_'+'DNS']
        der=FFT_derivative(var_vel,direction=deriv_dir,Fast=True)
        DL_model_outputs_DNS[var_name + '_' + 'DNS']=der
    
    
    # compute Q-R model 
    Model_DNS='Model'
    Q_f,R_f,S_ijS_ij_f,R_ijR_ij_f=Q_R_Calculator(DL_model_outputs_DNS['A11' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A12' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A13' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A21' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A22' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A23' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A31' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A32' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A33' + '_' + Model_DNS])
    DL_model_outputs_DNS['Q' + '_' + Model_DNS]=Q_f
    DL_model_outputs_DNS['R' + '_' + Model_DNS]=R_f
    DL_model_outputs_DNS['S_ijS_ij' + '_' + Model_DNS]=S_ijS_ij_f
    DL_model_outputs_DNS['R_ijR_ij' + '_' + Model_DNS]=R_ijR_ij_f
    
    # compute Q-R DNS 
    Model_DNS='DNS'
    Q_f,R_f,S_ijS_ij_f,R_ijR_ij_f=Q_R_Calculator(DL_model_outputs_DNS['A11' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A12' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A13' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A21' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A22' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A23' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A31' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A32' + '_' + Model_DNS],\
                                             DL_model_outputs_DNS['A33' + '_' + Model_DNS])
    DL_model_outputs_DNS['Q' + '_' + Model_DNS]=Q_f
    DL_model_outputs_DNS['R' + '_' + Model_DNS]=R_f
    DL_model_outputs_DNS['S_ijS_ij' + '_' + Model_DNS]=S_ijS_ij_f
    DL_model_outputs_DNS['R_ijR_ij' + '_' + Model_DNS]=R_ijR_ij_f
    
    
    
    dir_path='/'.join(path.split('/')[:-1])
    
    # plot PDF of velocity gradients
    plot_PDF_VelocityGrad_DL_Model_DNS(DL_model_outputs_DNS['A11' + '_' + 'DNS'],\
                                             DL_model_outputs_DNS['A12' + '_' + 'DNS'],\
                                             DL_model_outputs_DNS['A13' + '_' + 'DNS'],\
                                             DL_model_outputs_DNS['A21' + '_' + 'DNS'],\
                                             DL_model_outputs_DNS['A22' + '_' + 'DNS'],\
                                             DL_model_outputs_DNS['A23' + '_' + 'DNS'],\
                                             DL_model_outputs_DNS['A31' + '_' + 'DNS'],\
                                             DL_model_outputs_DNS['A32' + '_' + 'DNS'],\
                                             DL_model_outputs_DNS['A33' + '_' + 'DNS'],\
                                             DL_model_outputs_DNS['A11' + '_' + 'Model'],\
                                             DL_model_outputs_DNS['A12' + '_' + 'Model'],\
                                             DL_model_outputs_DNS['A13' + '_' + 'Model'],\
                                             DL_model_outputs_DNS['A21' + '_' + 'Model'],\
                                             DL_model_outputs_DNS['A22' + '_' + 'Model'],\
                                             DL_model_outputs_DNS['A23' + '_' + 'Model'],\
                                             DL_model_outputs_DNS['A31' + '_' + 'Model'],\
                                             DL_model_outputs_DNS['A32' + '_' + 'Model'],\
                                             DL_model_outputs_DNS['A33' + '_' + 'Model'],\
                                      dir_path=dir_path+'/VG_PDF.png')
    
    save_obj(DL_model_outputs_DNS, dir_path,'Dic_DL_model_outputs_DNS')
    #####
    
    return