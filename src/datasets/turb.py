import os
import torch
import h5py
import numpy as np
import joblib
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load


class Turb(Dataset):
    data_name = 'Turb'

    def __init__(self, root, split, subset):
        self.root = os.path.expanduser(root)
        self.split = split
        self.subset = subset
        if not check_exists(self.processed_folder):
            self.process()
        self.input, self.target = joblib.load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))

    def __getitem__(self, index):
        input, target = {s: torch.tensor(self.input[s][index]) for s in self.input}, \
                        {s: torch.tensor(self.target[s][index]) for s in self.target}
        return input

    def __len__(self):
        return len(self.input['Phy'])

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
        joblib.dump(train_set, os.path.join(self.processed_folder, 'train.pt'))
        joblib.dump(test_set, os.path.join(self.processed_folder, 'test.pt'))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nSubset: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.subset)
        return fmt_str

    def make_data(self):
        filenames = os.listdir(self.raw_folder)
        ts = []
        Phy = []
        for i in range(len(filenames)):
            filename_list = filenames[i].split('.')
            ts_i = int(filename_list[1])
            f = h5py.File('{}/{}'.format(self.raw_folder, filenames[i]), 'r')
            Phy_i = [f['Phy_U'][:], f['Phy_V'][:], f['Phy_W'][:]]
            Phy_i = np.stack(Phy_i, axis=0).astype(np.float32)
            ts.append(ts_i)
            Phy.append(Phy_i)
        num_train = round(len(ts) * 0.8)
        ts = np.array(ts, dtype=np.long)
        Phy = np.stack(Phy, axis=0)
        train_ts, test_ts = ts[:num_train], ts[num_train:]
        train_Phy, test_Phy = Phy[:num_train], Phy[num_train:]
        train_input = {'ts': train_ts[:-1], 'Phy': train_Phy[:-1]}
        train_target = {'ts': train_ts[1:], 'Phy': train_Phy[1:]}
        test_input = {'ts': test_ts[:-1], 'Phy': test_Phy[:-1]}
        test_target = {'ts': test_ts[1:], 'Phy': test_Phy[1:]}
        return (train_input, train_target), (test_input, test_target)