import torch
import datasets
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from config import cfg


def fetch_dataset(data_name, subset):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name == 'Turb':
        dataset['train'] = datasets.Turb(root=root, split='train', subset=subset)
        dataset['test'] = datasets.Turb(root=root, split='test', subset=subset)
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


def make_data_loader(dataset):
    data_loader = {}
    for k in dataset:
        data_loader[k] = torch.utils.data.DataLoader(dataset=dataset[k], shuffle=cfg['shuffle'][k],
                                                     batch_size=cfg['batch_size'][k], pin_memory=True,
                                                     num_workers=cfg['num_workers'], collate_fn=input_collate)
    return data_loader


class BatchDataset(Dataset):
    def __init__(self, dataset, seq_length):
        super().__init__()
        self.dataset = dataset
        self.seq_length = seq_length
        self.S = dataset[list(dataset.keys())[0]].size(1) if isinstance(self.dataset, dict) else dataset.size(1)
        self.idx = list(range(0, self.S - 1, seq_length))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        seq_length = min(self.seq_length, self.S - 1 - index)
        input = {'code': self.dataset[:, self.idx[index]:self.idx[index] + seq_length],
                 'ncode': self.dataset[:, self.idx[index] + seq_length]}
        return input