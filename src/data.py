import config
import torch
import datasets
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


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
        data_loader[k] = torch.utils.data.DataLoader(dataset=dataset[k], shuffle=config.PARAM['shuffle'][k],
                                                     batch_size=config.PARAM['batch_size'][k], pin_memory=True,
                                                     num_workers=config.PARAM['num_workers'], collate_fn=input_collate)
    return data_loader