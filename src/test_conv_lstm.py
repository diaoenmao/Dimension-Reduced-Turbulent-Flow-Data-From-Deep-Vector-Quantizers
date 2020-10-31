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
cfg['model_name'] = 'conv_lstm'

Use_cycling = False

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
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    ae = eval('models.{}().to(cfg["device"])'.format(cfg['ae_name']))
    _, ae, _, _, _ = resume(ae, cfg['ae_tag'], load_tag='best')
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    last_epoch, model, _, _, _ = resume(model, cfg['model_tag'], load_tag='best')
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    logger_path = 'output/runs/test_{}_{}'.format(cfg['model_tag'], current_time)
    test_logger = Logger(logger_path)
    test_logger.safe(True)
    if Use_cycling:
        test_cycle(dataset['test'], model, ae, test_logger, last_epoch)
    else:
        test(dataset['test'], model, ae, test_logger, last_epoch)
    test_logger.safe(False)
    _, _, _, _, train_logger = resume(model, cfg['model_tag'], load_tag='best')
    save_result = {'cfg': cfg, 'epoch': last_epoch, 'logger': {'train': train_logger, 'test': test_logger}}
    save(save_result, './output/result/{}.pt'.format(cfg['model_tag']))
    return

def test(dataset, model, ae, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        ae.train(False)
        model.train(False)
        Fast = False # true is memory consuming
        for i in range(len(dataset)-cfg['bptt']-1):
            
            # encode the input sequence 
            input_seq = [dataset[j] for j in range(i, i + cfg['bptt'])]
            input_seq = to_device(input_seq, cfg['device'])
            input_seq = [input_seq[j]['uvw'].unsqueeze(0) for j in range(len(input_seq))] 
            if Fast:
                input_seq = torch.cat(input_seq, dim=0)
                _, _, code = ae.encode(input_seq)
                code = code.unsqueeze(0)
            else:                
                code_data_i = []                
                for item in input_seq:
                    _, _, code_i = ae.encode(item)
                    code_data_i.append(code_i)                                
                code = torch.stack(code_data_i, dim=1)             
            
            # encode the target 
            input = dataset[i + cfg['bptt']] # target
            input_size = input['uvw'].size(0)
            input = to_device(input, cfg['device'])
            input['uvw'], input['duvw'] = input['uvw'].unsqueeze(0), input['duvw'].unsqueeze(0) # add batch dimension
            input['code'] = code
            _, _, input['ncode'] = ae.encode(input['uvw'])
            output = model(input)            
            output['uvw'] = ae.decode_code(output['code'])            
            output['duvw'] = models.spectral_derivative_3d(output['uvw'])
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test'], input, output)
            print([(key,evaluation[key]) for key in evaluation])
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test'])
        vis(input, output, './output/vis')
    return


def test_cycle(dataset, model, ae, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        ae.train(False)
        model.train(False)
        
        input = [dataset[j] for j in range(cfg['bptt'])]
        input = to_device(input, cfg['device'])
        initial_seq = [input[j]['uvw'].unsqueeze(0) for j in range(len(input))]
        code_data_i = []                
        for item in initial_seq:
            _, _, code_i = ae.encode(item)
            code_data_i.append(code_i)                                
        code = torch.stack(code_data_i, dim=1)                
        for i in range(cfg['bptt'],len(dataset)):
            input = dataset[i] # target
            input_size = input['uvw'].size(0)
            input = to_device(input, cfg['device'])
            input['uvw'], input['duvw'] = input['uvw'].unsqueeze(0), input['duvw'].unsqueeze(0) # add batch dimension
            input['code'] = code
            _, _, input['ncode'] = ae.encode(input['uvw'])
            output = model(input)            
            output['uvw'] = ae.decode_code(output['code'])            
            output['duvw'] = models.spectral_derivative_3d(output['uvw'])
            # update input code by shifting the current code (code[:-1]=code[1:]) and using the new predicted code                        
            code[:,:-1] = code[:,1:].clone()
            code[:,-1] = output['code']                                   
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test'], input, output)
            logger.append(evaluation, 'test', input_size)
            vis(input, output, './output/vis/prediction{}'.format(i+1-cfg['bptt']))
            print([(key,evaluation[key]) for key in evaluation])
            if i==2*cfg['bptt']:
                break
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test'])
        #vis(input, output, './output/vis')
    return
if __name__ == "__main__":
    main()