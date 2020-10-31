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
cfg['metric_name'] = {'train': ['Loss'], 'test': ['Loss', 'MSE']}
cfg['ae_name'] = 'vqvae'
cfg['model_name'] = 'transformer'


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        ae_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['ae_name'], cfg['control_name']]
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        cfg['ae_tag'] = '_'.join([x for x in ae_tag_list if x])
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    uvw_dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    code_dataset = {}
    code_dataset['test'] = load('./output/code/test_{}.pt'.format(cfg['ae_tag']))
    ae = eval('models.{}().to(cfg["device"])'.format(cfg['ae_name']))
    _, ae, _, _, _ = resume(ae, cfg['ae_tag'], load_tag='best')
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    last_epoch, model, _, _, _ = resume(model, cfg['model_tag'], load_tag='best')
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    logger_path = 'output/runs/test_{}_{}'.format(cfg['model_tag'], current_time)
    test_logger = Logger(logger_path)
    test_logger.safe(True)
    test(uvw_dataset['test'], code_dataset['test'], model, ae, test_logger, last_epoch)
    test_logger.safe(False)
    _, _, _, _, train_logger = resume(model, cfg['model_tag'], load_tag='best')
    save_result = {'cfg': cfg, 'epoch': last_epoch, 'logger': {'train': train_logger, 'test': test_logger}}
    save(save_result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def test(uvw_dataset, code_dataset, model, ae, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        ae.train(False)
        model.train(False)
        for i in range(0, len(uvw_dataset) - 2 * cfg['bptt'], 1):
            input_uvw, input_duvw = [], []
            for j in range(i, i + 2 * cfg['bptt']):
                input_uvw.append(uvw_dataset[j]['uvw'])
                input_duvw.append(uvw_dataset[j]['duvw'])
            code = code_dataset[i: i + 2 * cfg['bptt']]
            input_uvw = torch.stack(input_uvw, dim=0)
            input_duvw = torch.stack(input_duvw, dim=0)
            code = code.unsqueeze(0)
            input = {'uvw': input_uvw[cfg['bptt']:], 'duvw': input_duvw[cfg['bptt']:],
                     'code': code[:, :cfg['bptt']], 'ncode': code[:, cfg['bptt']:]}
            input = to_device(input, cfg['device'])
            output = model(input)
            output['uvw'] = ae.decode_code(output['code'].view(-1, *output['code'].size()[2:]))
            output['duvw'] = models.spectral_derivative_3d(output['uvw'])
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test'], input, output)
            logger.append(evaluation, 'test', 1)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test'])
        vis(input, output, './output/vis')
    return


if __name__ == "__main__":
    main()