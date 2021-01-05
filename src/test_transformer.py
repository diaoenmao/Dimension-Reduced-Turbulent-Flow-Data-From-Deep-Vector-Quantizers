import argparse
import datetime
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg
from data import BatchDataset, fetch_dataset
from metrics import Metric
from utils import save, load, to_device, process_control, process_dataset, resume, vis
from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']]) if 'control' in cfg else ''
cfg['metric_name'] = {'train': ['Loss'], 'test': ['Loss', 'MSE', 'D_MSE', 'Physics']}
cfg['ae_name'] = 'vqvae'


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        cfg['ae_control_name'] = '_'.join([cfg['control'][k] for k in cfg['control'] if k not in ['seq_length']])
        ae_tag_list = [str(seeds[i]), cfg['data_name'], cfg['ae_name'], cfg['ae_control_name']]
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['ae_tag'] = '_'.join([x for x in ae_tag_list if x])
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    uvw_dataset = fetch_dataset(cfg['data_name'])
    code_dataset = {}
    code_dataset['test'] = load('./output/code/test_{}.pt'.format(cfg['ae_tag']))
    ae = eval('models.{}().to(cfg["device"])'.format(cfg['ae_name']))
    _, ae, _, _, _ = resume(ae, cfg['ae_tag'], load_tag='best')
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    metric = Metric({'test': ['MSE', 'D_MSE', 'Physics']})
    last_epoch, model, _, _, _ = resume(model, cfg['model_tag'], load_tag='best')
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    logger_path = 'output/runs/test_{}_{}'.format(cfg['model_tag'], current_time)
    test_logger = Logger(logger_path)
    test_logger.safe(True)
    test(uvw_dataset['test'], code_dataset['test'], ae, model, metric, test_logger, last_epoch)
    test_logger.safe(False)
    _, _, _, _, train_logger = resume(model, cfg['model_tag'], load_tag='best')
    save_result = {'cfg': cfg, 'epoch': last_epoch, 'logger': {'train': train_logger, 'test': test_logger}}
    save(save_result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def test(uvw_dataset, code_dataset, ae, model, metric, logger, epoch):
    with torch.no_grad():
        ae.train(False)
        model.train(False)
        for i in range(0, len(uvw_dataset) - (cfg['seq_length'][0] + cfg['seq_length'][1])):
            uvw, duvw = [], []
            for j in range(i + cfg['seq_length'][0], i + cfg['seq_length'][0] + cfg['seq_length'][1]):
                uvw.append(uvw_dataset[j]['uvw'])
                duvw.append(uvw_dataset[j]['duvw'])
            uvw = torch.stack(uvw, dim=0)
            duvw = torch.stack(duvw, dim=0)
            code = code_dataset[i: i + cfg['seq_length'][0]].unsqueeze(0)
            ncode = model.next(code.to(cfg['device']), cfg['seq_length'][1])
            input = {'uvw': uvw, 'duvw': duvw, 'code': code}
            output = {'ncode': ncode}
            input = to_device(input, cfg['device'])
            output['uvw'] = ae.decode_code(output['ncode'].view(-1, *output['ncode'].size()[2:]))
            output['duvw'] = models.spectral_derivative_3d(output['uvw'])
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', 1)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
        vis(input, output, './output/vis')
    return


if __name__ == "__main__":
    main()
