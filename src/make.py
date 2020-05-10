import argparse
import config

config.init()
import itertools

parser = argparse.ArgumentParser(description='Config')
parser.add_argument('--run', default=None, type=str)
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--file', default=None, type=str)
args = vars(parser.parse_args())


def main():
    round = 12
    run = args['run']
    model = args['model']
    file = model if args['file'] is None else args['file']
    filename = '{}_{}'.format(run, file)
    gpu_ids = ['0', '1', '2', '3']
    script_name = [['{}_{}.py'.format(run, file)]]
    data_names = ['Turb']
    model_names = [['unet']]
    experiments_step = 1
    num_experiments = 1
    init_seeds = [list(range(0, num_experiments, experiments_step))]
    num_epochs = [[200]]
    num_experiments = [[experiments_step]]
    s = '#!/bin/bash\n'
    k = 0
    for i in range(len(data_names)):
        data_name = data_names[i]
        controls = script_name + [[data_name]] + model_names + init_seeds + num_epochs + num_experiments
        controls = list(itertools.product(*controls))
        for j in range(len(controls)):
            controls[j] = list(controls[j])
            if 'mc' in controls[j][2]:
                controls[j].append('0.5')
            else:
                controls[j].append('None')
            s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --data_name {} --model_name {} --init_seed {} ' \
                    '--num_epochs {} --num_experiments {} --control_name {}&\n'.format(
                gpu_ids[k % len(gpu_ids)], *controls[j])
            if j % round == round - 1:
                s = s[:-2] + '\n'
            k = k + 1
    print(s)
    run_file = open('./{}.sh'.format(filename), 'w')
    run_file.write(s)
    run_file.close()
    exit()


if __name__ == '__main__':
    main()