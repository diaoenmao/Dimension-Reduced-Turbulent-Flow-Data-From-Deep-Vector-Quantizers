import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load


class Turb(Dataset):
    data_name = 'Turb'

    def __init__(self, root, split):
        self.root = os.path.expanduser(root)
        self.split = split
        if not check_exists(self.processed_folder):
            self.process()
        self.data = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))

    def __getitem__(self, index):
        data = {}
        for s in self.data:
            if isinstance(self.data[s][index], str):
                data[s] = torch.tensor(load(self.data[s][index]))
            else:
                data[s] = torch.tensor(self.data[s][index])
        return data

    def __len__(self):
        return len(self.data['uvw'])

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            raise ValueError('Not valid dataset')
        train_set, test_set = self.make_data()
        makedir_exist_ok(self.processed_folder)
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(self.__class__.__name__, self.__len__(), self.root,
                                                                     self.split)
        return fmt_str

    def make_data(self):
        sub_folder = 'Data_Re_90_Fr_Inf_Ng_128_Npr_16_AcD_V25'
        filename_lead = 'V25_Phy_Vel_VelG'
        Ng, Nb = 128, 16
        train_ts = np.arange(4050, 7050, 75).astype(np.int64)
        test_ts = np.arange(7050, 10050, 75).astype(np.int64)
        train_uvw = []
        train_duvw = []
        for i in range(len(train_ts)):
            u, v, w = np.zeros((Ng, Ng, Ng)), np.zeros((Ng, Ng, Ng)), np.zeros((Ng, Ng, Ng))
            du, dv, dw = np.zeros((3, Ng, Ng, Ng)), np.zeros((3, Ng, Ng, Ng)), np.zeros((3, Ng, Ng, Ng))
            for b in range(0, Nb):
                f = h5py.File(os.path.join(self.raw_folder, sub_folder, '{}.{:06.0f}.h5.{:06.0f}'.format(
                    filename_lead, train_ts[i], b)), 'r')
                i_min, i_max = b // 4 * Ng // 4, (b // 4 + 1) * Ng // 4
                j_min, j_max = (b % 4) * Ng // 4, ((b % 4) + 1) * Ng // 4
                u[i_min:i_max, j_min:j_max, :] = f['Phy_W'][:]
                v[i_min:i_max, j_min:j_max, :] = f['Phy_V'][:]
                w[i_min:i_max, j_min:j_max, :] = f['Phy_U'][:]
                du[0, i_min:i_max, j_min:j_max, :] = f['dWdzG'][:]
                du[1, i_min:i_max, j_min:j_max, :] = f['dWdyG'][:]
                du[2, i_min:i_max, j_min:j_max, :] = f['dWdxG'][:]
                dv[0, i_min:i_max, j_min:j_max, :] = f['dVdzG'][:]
                dv[1, i_min:i_max, j_min:j_max, :] = f['dVdyG'][:]
                dv[2, i_min:i_max, j_min:j_max, :] = f['dVdxG'][:]
                dw[0, i_min:i_max, j_min:j_max, :] = f['dUdzG'][:]
                dw[1, i_min:i_max, j_min:j_max, :] = f['dUdyG'][:]
                dw[2, i_min:i_max, j_min:j_max, :] = f['dUdxG'][:]
                f.close()
            uvw = np.stack([u, v, w], axis=0).astype(np.float32)
            duvw = np.stack([du, dv, dw], axis=0).astype(np.float32)
            save(uvw, os.path.join(self.raw_folder, '{}.pkl'.format(train_ts[i])))
            save(duvw, os.path.join(self.raw_folder, '{}_d.pkl'.format(train_ts[i])))
            train_uvw.append(os.path.join(self.raw_folder, '{}.pkl'.format(train_ts[i])))
            train_duvw.append(os.path.join(self.raw_folder, '{}_d.pkl'.format(train_ts[i])))
        test_uvw = []
        test_duvw = []
        for i in range(len(test_ts)):
            u, v, w = np.zeros((Ng, Ng, Ng)), np.zeros((Ng, Ng, Ng)), np.zeros((Ng, Ng, Ng))
            du, dv, dw = np.zeros((3, Ng, Ng, Ng)), np.zeros((3, Ng, Ng, Ng)), np.zeros((3, Ng, Ng, Ng))
            for b in range(0, Nb):
                f = h5py.File(os.path.join(self.raw_folder, sub_folder, '{}.{:06.0f}.h5.{:06.0f}'.format(
                    filename_lead, test_ts[i], b)), 'r')
                i_min, i_max = b // 4 * Ng // 4, (b // 4 + 1) * Ng // 4
                j_min, j_max = (b % 4) * Ng // 4, ((b % 4) + 1) * Ng // 4
                u[i_min:i_max, j_min:j_max, :] = f['Phy_W'][:]
                v[i_min:i_max, j_min:j_max, :] = f['Phy_V'][:]
                w[i_min:i_max, j_min:j_max, :] = f['Phy_U'][:]
                du[0, i_min:i_max, j_min:j_max, :] = f['dWdzG'][:]
                du[1, i_min:i_max, j_min:j_max, :] = f['dWdyG'][:]
                du[2, i_min:i_max, j_min:j_max, :] = f['dWdxG'][:]
                dv[0, i_min:i_max, j_min:j_max, :] = f['dVdzG'][:]
                dv[1, i_min:i_max, j_min:j_max, :] = f['dVdyG'][:]
                dv[2, i_min:i_max, j_min:j_max, :] = f['dVdxG'][:]
                dw[0, i_min:i_max, j_min:j_max, :] = f['dUdzG'][:]
                dw[1, i_min:i_max, j_min:j_max, :] = f['dUdyG'][:]
                dw[2, i_min:i_max, j_min:j_max, :] = f['dUdxG'][:]
                f.close()
            uvw = np.stack([u, v, w], axis=0).astype(np.float32)
            duvw = np.stack([du, dv, dw], axis=0).astype(np.float32)
            save(uvw, os.path.join(self.raw_folder, '{}.pkl'.format(test_ts[i])))
            save(duvw, os.path.join(self.raw_folder, '{}_d.pkl'.format(test_ts[i])))
            test_uvw.append(os.path.join(self.raw_folder, '{}.pkl'.format(test_ts[i])))
            test_duvw.append(os.path.join(self.raw_folder, '{}_d.pkl'.format(test_ts[i])))
        train_uvw_result = {'ts': train_ts, 'uvw': train_uvw, 'duvw': train_duvw}
        test_uvw_result = {'ts': test_ts, 'uvw': test_uvw, 'duvw': test_duvw}
        return train_uvw_result, test_uvw_result
