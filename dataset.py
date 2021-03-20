import os
import numpy as np
import random
from PIL import Image
import data_transforms as transforms
import torch


class DehazeList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, out_name=False):
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.out_name = out_name

        self.image_list = None
        self.gt_list = None
        self.name_list = None
        self._make_list()

    def __getitem__(self, index):
        np.random.seed()
        random.seed()
        image = Image.open(os.path.join(self.data_dir, self.phase, self.image_list[index]))
        data = [image]

        if self.gt_list is not None:
            gt = Image.open(os.path.join(self.data_dir, self.phase, self.gt_list[index]))
            data.append(gt)
 
        data = list(self.transforms(*data))

        if self.out_name:
            data.append(os.path.basename(self.image_list[index]))

        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def _make_list(self):
        # load paths of images from txt files
        image_path = os.path.join('./datasets', self.phase + '_image.txt')
        gt_path = os.path.join('./datasets', self.phase + '_gt.txt')
        assert os.path.exists(image_path)
        
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        
        if os.path.exists(gt_path):
            self.gt_list = [line.strip() for line in open(gt_path, 'r')]
            assert len(self.image_list) == len(self.gt_list)
        