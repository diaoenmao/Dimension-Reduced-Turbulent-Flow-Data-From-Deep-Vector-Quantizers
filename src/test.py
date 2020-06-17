import argparse
import datetime
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg
from data import fetch_dataset, make_data_loader
from utils import save, load, to_device, process_control, process_dataset, resume, collate
from logger import Logger

if __name__ == "__main__":
    process_control()
    dataset = fetch_dataset('Turb', subset='uvw')
    data_loader = make_data_loader(dataset)
    input = next(iter(data_loader['train']))
    input = collate(input)
    print(input['ts'].size(), input['uvw'].size())
    print(input['ts'].dtype, input['uvw'].dtype)
