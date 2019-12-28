import config

config.init()
import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control_name, process_dataset, resume, collate
from logger import Logger

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Config')
for k in config.PARAM:
    exec('parser.add_argument(\'--{0}\',default=config.PARAM[\'{0}\'], type=type(config.PARAM[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in config.PARAM:
    config.PARAM[k] = args[k]
if args['control_name']:
    config.PARAM['control_name'] = args['control_name']
    control_list = list(config.PARAM['control'].keys())
    control_name_list = args['control_name'].split('_')
    for i in range(len(control_name_list)):
        config.PARAM['control'][control_list[i]] = control_name_list[i]
control_name_list = []
for k in config.PARAM['control']:
    control_name_list.append(config.PARAM['control'][k])
config.PARAM['control_name'] = '_'.join(control_name_list)


def main():
    process_control_name()
    seeds = list(range(config.PARAM['init_seed'], config.PARAM['init_seed'] + config.PARAM['num_Experiments']))
    for i in range(config.PARAM['num_Experiments']):
        model_tag_list = [str(seeds[i]), config.PARAM['data_name'], config.PARAM['subset'], config.PARAM['model_name'],
                          config.PARAM['control_name']]
        config.PARAM['model_tag'] = '_'.join(filter(None, model_tag_list))
        print('Experiment: {}'.format(config.PARAM['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(config.PARAM['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(config.PARAM['data_name'], config.PARAM['subset'])
    process_dataset(dataset['train'])
    data_loader = make_data_loader(dataset)
    model = eval('models.{}().to(config.PARAM["device"])'.format(config.PARAM['model_name']))
    if config.PARAM['world_size'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(config.PARAM['world_size'])))
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)
    if config.PARAM['resume_mode'] == 1:
        last_epoch, model, optimizer, scheduler, logger = resume(model, config.PARAM['model_tag'], optimizer, scheduler)
    elif config.PARAM['resume_mode'] == 2:
        last_epoch = 1
        _, model, _, _, _ = resume(model, config.PARAM['model_tag'])
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/{}_{}'.format(config.PARAM['model_tag'], current_time)
        logger = Logger(logger_path)
    else:
        last_epoch = 1
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/train_{}_{}'.format(config.PARAM['model_tag'], current_time) if config.PARAM[
            'log_overwrite'] else 'output/runs/train_{}'.format(config.PARAM['model_tag'])
        logger = Logger(logger_path)
    config.PARAM['pivot_metric'] = 'test/Loss'
    config.PARAM['pivot'] = 1e10
    for epoch in range(last_epoch, config.PARAM['num_epochs'] + 1):
        logger.safe(True)
        train(data_loader['train'], model, optimizer, logger, epoch)
        test(data_loader['test'], model, logger, epoch)
        if config.PARAM['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler.step(metrics=logger.tracker[config.PARAM['pivot_metric']], epoch=epoch)
        else:
            scheduler.step(epoch=epoch + 1)
        if config.PARAM['save_mode'] >= 0:
            logger.safe(False)
            model_state_dict = model.module.state_dict() if config.PARAM['world_size'] > 1 else model.state_dict()
            save_result = {
                'config': config.PARAM, 'epoch': epoch + 1, 'model_dict': model_state_dict,
                'optimizer_dict': optimizer.state_dict(), 'scheduler_dict': scheduler.state_dict(),
                'logger': logger}
            save(save_result, './output/model/{}_checkpoint.pt'.format(config.PARAM['model_tag']))
            if config.PARAM['pivot'] > logger.tracker[config.PARAM['pivot_metric']]:
                config.PARAM['pivot'] = logger.tracker[config.PARAM['pivot_metric']]
                shutil.copy('./output/model/{}_checkpoint.pt'.format(config.PARAM['model_tag']),
                            './output/model/{}_best.pt'.format(config.PARAM['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(data_loader, model, optimizer, logger, epoch):
    metric = Metric()
    model.train(True)
    for i, input in enumerate(data_loader):
        start_time = time.time()
        input = collate(input)
        input_size = len(input['A'])
        input = to_device(input, config.PARAM['device'])
        model.zero_grad()
        output = model(input)
        output['loss'] = output['loss'].mean() if config.PARAM['world_size'] > 1 else output['loss']
        output['loss'].backward()
        optimizer.step()
        if i % int((len(data_loader) * config.PARAM['log_interval']) + 1) == 0:
            batch_time = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((config.PARAM['num_epochs'] - epoch) * batch_time * len(data_loader)))
            info = {'info': ['Model: {}'.format(config.PARAM['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                             'Learning rate: {}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            evaluation = metric.evaluate(config.PARAM['metric_names']['train'], input, output)
            logger.append(evaluation, 'train', n=input_size)
            logger.write('train', config.PARAM['metric_names']['train'])
    return


def test(data_loader, model, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = len(input['A'])
            input = to_device(input, config.PARAM['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if config.PARAM['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(config.PARAM['metric_names']['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(config.PARAM['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', config.PARAM['metric_names']['test'])
    return


def make_optimizer(model):
    if config.PARAM['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.PARAM['lr'], momentum=config.PARAM['momentum'],
                              weight_decay=config.PARAM['weight_decay'])
    elif config.PARAM['optimizer_name'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config.PARAM['lr'], momentum=config.PARAM['momentum'],
                                  weight_decay=config.PARAM['weight_decay'])
    elif config.PARAM['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.PARAM['lr'], weight_decay=config.PARAM['weight_decay'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer):
    if config.PARAM['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif config.PARAM['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.PARAM['step_size'],
                                              gamma=config.PARAM['factor'])
    elif config.PARAM['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.PARAM['milestones'],
                                                   gamma=config.PARAM['factor'])
    elif config.PARAM['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.PARAM['num_epochs'])
    elif config.PARAM['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.PARAM['factor'],
                                                         patience=config.PARAM['patience'], verbose=True,
                                                         threshold=config.PARAM['threshold'],
                                                         threshold_mode='rel')
    elif config.PARAM['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.PARAM['lr'], max_lr=10 * config.PARAM['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


if __name__ == "__main__":
    main()