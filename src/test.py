import config

config.init()
import argparse
import datetime
import torch
import torch.backends.cudnn as cudnn
import models
from data import fetch_dataset, make_data_loader
from utils import save, load, to_device, process_control_name, process_dataset, resume, collate
from logger import Logger

if __name__ == "__main__":
    process_control_name()
    dataset = fetch_dataset('TURB', subset='Phy')
    data_loader = make_data_loader(dataset)
    model = models.cascade()
    input = next(iter(data_loader['train']))
    input = collate(input)
    print(input['Phy'].size(), input['Phy'].size())
    output = model(input)