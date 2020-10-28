import argparse
import datetime
import os
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import models
from config import cfg
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, load, to_device, process_control, process_dataset, resume, collate, vis
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
cfg['metric_name'] = {'train': ['Loss', 'MSE', 'D_MSE', 'Physics'], 'test': ['Loss', 'MSE', 'D_MSE', 'Physics']}
cfg['ae_name'] = 'vqvae'
cfg['prediction_name'] = 'conv_lstm'
cfg['model_name'] = cfg['prediction_name']
cfg['data_shape'] = [3, 128, 128, 128]

def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_ae_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['ae_name'], 'code'+str(128//2**cfg['vqvae']['depth']), cfg['control_name']]
        model_prediction_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['prediction_name'], *model_ae_tag_list[3:5], cfg['control_name']]
        cfg['model_ae_tag'] = '_'.join([x for x in model_ae_tag_list if x])
        cfg['model_prediction_tag'] = '_'.join([x for x in model_prediction_tag_list if x])
        print('Experiment: ae :  {} , prediction :  {}'.format(cfg['model_ae_tag'], cfg['model_prediction_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_prediction_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    """
    code_dataset = {'train' : {}, 'test' : {}}
    code_dataset['train']['code'] = load('./output/code/train_{}.pt'.format(cfg['model_ae_tag']))
    code_dataset['train']['quantized'] = load('./output/quantized/train_{}.pt'.format(cfg['model_ae_tag']))
    code_dataset['test']['code'] = load('./output/code/test_{}.pt'.format(cfg['model_ae_tag']))
    code_dataset['test']['quantized'] = load('./output/quantized/test_{}.pt'.format(cfg['model_ae_tag']))
    print(code_dataset['train']['quantized'].size(), "     " , code_dataset['train']['code'].size())
    process_dataset(code_dataset)
    print(code_dataset['train']['quantized'].size(), "     " , code_dataset['train']['code'].size())
    """
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    data_loader = dataset #make_data_loader(dataset)
    trained_models = [None, None]
    
    trained_models[0] = eval('models.{}().to(cfg["device"])'.format(cfg['ae_name']))
    trained_models[1] = eval('models.{}().to(cfg["device"])'.format(cfg['prediction_name']))
    load_tag = 'best'
    last_epoch, trained_models[0], _, _, _ = resume(trained_models[0], cfg['model_ae_tag'], load_tag=load_tag)
    last_epoch, trained_models[1], _, _, _ = resume(trained_models[1], cfg['model_prediction_tag'], load_tag=load_tag)
    
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    logger_path = 'output/runs/test_{}_{}'.format(cfg['model_prediction_tag'], current_time)
    logger = Logger(logger_path)
    logger.safe(True)
    
    test(data_loader['train'], trained_models, logger, last_epoch)
    logger.safe(False)
    #save_result = {'config': cfg, 'epoch': last_epoch, 'logger': logger}
    #save(save_result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def test(data_loader, model, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        model[0].train(False)
        model[1].train(False)
        pred=[]
        print(len(data_loader))
        print(data_loader[0].keys())
        print(data_loader[0]['uvw'].size())
        SeqL = cfg['bptt']
        
        for i in range(len(data_loader)-SeqL-1):
            input = [data_loader[j] for j in range(i, i+SeqL+1)]
            print("i = " , i , " len(input) = ", len(input))
            input_size = input[0]['uvw'].size(0)
            input = to_device(input, cfg['device']) #print(input[0]['uvw'].size())            
            
            if i==0:                
                x = [input[j]['uvw'].unsqueeze(0) for j in range(SeqL)]
                pred.extend(x)
                # encode x
                code_x = []
                quantized_x = []
                for item in x:
                    quantized_i, _, code_i = model[0].encode(item)
                    code_x.append(code_i)
                    quantized_x.append(quantized_i)
                quantized_x = torch.stack(quantized_x, dim=1)
                code_x = torch.stack(code_x, dim=1)
                #print(" code_x.size() = ", code_x.size()), " quantized_x.size() = ", quantized_x.size())
                
            y = [input[j]['uvw'].unsqueeze(0) for j in range(1, SeqL+1)]            
            code_y = []
            quantized_y = []
            for item in y:
                quantized_i, _, code_i = model[0].encode(item)
                code_y.append(code_i)
                quantized_y.append(quantized_i)
            
            quantized_y = torch.stack(quantized_y, dim=1)
            code_y = torch.stack(code_y, dim=1)
            
            lstm_input = {'code' : code_x, 'quantized' : quantized_x, 'ncode' : code_y, 'nquantized' : quantized_y}
            
            output = model[1](lstm_input)
            #print("output['q_score'].size() = ", output['q_score'].size())
            new_pred = model[0].decode(output['q_score'][:,-1]) # last element
            #print("new_pred.size() = ", new_pred.size())
            print("MSE = ", F.mse_loss(new_pred , y[-1]) )
            pred.append(new_pred)
            
            code_x = code_y
            quantized_x = quantized_y
            
            """
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        logger.append(evaluation, 'test')
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test'])
        #vis(input, output, './output/vis')"""
    return


if __name__ == "__main__":
    main()