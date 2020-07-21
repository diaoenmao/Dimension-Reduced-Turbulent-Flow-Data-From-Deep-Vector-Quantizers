import argparse
import itertools

parser = argparse.ArgumentParser(description='Config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--num_gpu', default=4, type=int)
parser.add_argument('--experiments_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
args = vars(parser.parse_args())


def main():
    run = args['run']
    model = args['model']
    round = args['round']
    num_gpu = args['num_gpu']
    experiments_step = args['experiments_step']
    num_experiments = args['num_experiments']
    num_epochs = args['num_epochs']
    resume_mode = args['resume_mode']
    gpu_ids = [str(x) for x in list(range(num_gpu))]
    if run in ['train', 'test']:
        filename = '{}_{}'.format(run, model)
        script_name = [['{}_{}.py'.format(run, model)]]
    else:
        filename = '{}_{}'.format(run, model)
        script_name = [['{}.py'.format(run)]]
    data_names = ['Turb']
    model_names = [[model]]
    init_seeds = [list(range(0, num_experiments, experiments_step))]
    num_epochs = [[num_epochs]]
    resume_mode = [[resume_mode]]
    num_experiments = [[experiments_step]]
    control = [['1', '2', '3', '4', '5', '6']]
    s = '#!/bin/bash\n'
    k = 0
    for i in range(len(data_names)):
        data_name = data_names[i]
        controls = script_name + [[data_name]] + model_names + init_seeds + num_experiments + num_epochs + \
                   resume_mode + control
        controls = list(itertools.product(*controls))
        for j in range(len(controls)):
            controls[j] = list(controls[j])
            s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --data_name {} --model_name {} --init_seed {} ' \
                    '--num_experiments {} --num_epochs {} --resume_mode {} --control_name {}&\n'.format(
                gpu_ids[k % len(gpu_ids)], *controls[j])
            if k % round == round - 1:
                s = s[:-2] + '\n'
            k = k + 1
    print(s)
    run_file = open('./{}.sh'.format(filename), 'w')
    run_file.write(s)
    run_file.close()
    return


if __name__ == '__main__':
    main()