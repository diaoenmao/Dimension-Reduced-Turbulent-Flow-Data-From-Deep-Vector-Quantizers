import torch
import numpy as np
import datasets
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from config import cfg


def fetch_dataset(data_name):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name == 'Turb':
        dataset['train'] = datasets.Turb(root=root, split='train')
        dataset['test'] = datasets.Turb(root=root, split='test')
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, shuffle=None):
    data_loader = {}
    for k in dataset:
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        data_loader[k] = DataLoader(dataset=dataset[k], shuffle=_shuffle, batch_size=cfg[tag]['batch_size'][k],
                                    pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                    worker_init_fn=np.random.seed(0))
    return data_loader


class BatchDataset(Dataset):
    def __init__(self, dataset, seq_length):
        super().__init__()
        self.dataset = dataset
        self.seq_length = seq_length
        self.S = dataset.size(1)
        self.idx = list(range(0, self.S - (self.seq_length[0] + self.seq_length[1])))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        input = {'code': self.dataset[:, self.idx[index]:self.idx[index] + self.seq_length[0]],
                 'ncode': self.dataset[:, self.idx[index] + self.seq_length[0]:self.idx[index] +
                                                                               self.seq_length[0] + self.seq_length[1]]}
        return input
