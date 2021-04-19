import argparse
import logging
import os
import threading
import time
import numpy as np
import itertools
from train import train_dehaze
from test import test_dehaze

import sys
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import datetime
from dataset import DehazeList
import data_transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description='')

    # Mode selection
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('--phase', default='val')

    # System parameters
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--gpus', type=str, default='', nargs='+', help='decides gpu(s) to use(ex. \'python main.py --gpus 0 1\'')

    # File loading/saving parameters
    parser.add_argument('--update-fileli', type=str, default='true', help='decides to update image file list text file or not(default: true)')
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint of model (default: none)')
    parser.add_argument('--name', default='noname', type=str, help='name of the experiment')

    # Data augmentation parameters
    parser.add_argument('-s', '--crop-size', default=0, type=int) 
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--random-identity-mapping', default=0, type=float)
    
    # Training parameters
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default=None)
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--coeff-l1', '--cl1', default=1.0, type=float,
                        metavar='CL1', help='coefficient of l1 loss (default: 1)')
    parser.add_argument('--coeff-cl', '--cl', default=0.5, type=float,
                        metavar='CL', help='coefficient of content loss (default: 0.5)')
    parser.add_argument('--coeff-ssim', '--ssim', default=1e-2, type=float,
                        metavar='SSIM', help='coefficient of ssim loss (default: )')
        
    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)

    return args


def make_data_txt(data_dir, list_dir = './datasets'):
    list_dir = list_dir
    # make the directory to save text files if it doesn't exist
    if not os.path.exists(list_dir):
        os.makedirs(list_dir, exist_ok=True)
    
    phase_list = ['train', 'val', 'test']
    img_type_list = list(zip(['image', 'gt'], ['HAZY', 'GT']))

    for phase, img_type in itertools.product(phase_list, img_type_list):
        dir = os.path.join(data_dir, phase, img_type[1])
        if os.path.exists(dir):
            f = open(os.path.join(list_dir, phase + '_' + img_type[0] + '.txt'), 'w')
            img_list = [os.path.join(img_type[1], img) \
                            for img in os.listdir(dir) if (img.endswith('png') or img.endswith('jpg'))]
            img_list.sort()

            for _ in range(1):
                for item in img_list:
                    f.write(item + '\n')
                if phase != 'train':
                    break
            
            f.close()
    
    
def main():
    args = parse_args()
    
    # setting the gpus to use in training/testing
    os.environ['CUDA_VISIBLE_DEVICES'] = "".join(args.gpus)
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    # Update image file list
    if args.update_fileli.lower() == 'true':
        make_data_txt(data_dir=args.data_dir)
    elif args.update_fileli.lower() == 'false':
        pass
    
    # Logging configuration
    save_dir = './checkpoints/' + args.name
    if not os.path.exists(save_dir) and args.cmd != 'test':
        os.makedirs(save_dir, exist_ok=True)

    FORMAT = "[%(asctime)-15s %(filename)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if args.cmd == 'train':
        if args.resume:
            file_handler = logging.FileHandler(save_dir + '/log_training.log', mode='a')
        else:
            file_handler = logging.FileHandler(save_dir + '/log_training.log', mode='w')
        logger.addHandler(file_handler)
        train_dehaze(args, save_dir=save_dir, logger=logger)
    elif args.cmd == 'test':
        test_dehaze(args, save_dir=os.path.dirname(args.resume), logger=logger, logging=logging)

if __name__ == '__main__':
    main()
