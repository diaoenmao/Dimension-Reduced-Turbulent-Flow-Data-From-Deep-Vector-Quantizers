import os
from config import cfg
from data import fetch_dataset, make_data_loader
from utils import save, collate, Stats, makedir_exist_ok, process_dataset, process_control

if __name__ == "__main__":
    stats_path = './res/stats'
    dim = 1
    data_names = ['Turb']
    process_control()
    cfg['seed'] = 0
    cfg['tag'] = {'batch_size': {'train': 1, 'test': 1}}
    for data_name in data_names:
        dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
        process_dataset(dataset['train'])
        data_loader = make_data_loader(dataset)
        stats = Stats(dim=dim)
        for i, input in enumerate(data_loader['train']):
            input = collate(input)
            stats.update(input['uvw'])
        stats = (stats.mean.tolist(), stats.std.tolist())
        print(data_name, stats)
        makedir_exist_ok(stats_path)
        save(stats, os.path.join(stats_path, '{}.pt'.format(data_name)))
