# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:45:32 2021

@author: GJMY
"""

import os
# import random
import numpy as np
# import nibabel as nib
from torch.utils.data import Dataset
from PIL import Image

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir,self.split+'.txt')).readlines()
        self.data_dir = base_dir
        
        self.img_ids = [i_id.strip() for i_id in self.sample_list]
        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(list_dir, "data\case_img_%s.png" % name)
            label_file = os.path.join(list_dir, "label\case_label_%s.png" % name)
            self.files.append((img_file,label_file))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_file,label_file = self.files[idx]
        '''load the datas'''
        
        

        return np.float32(Image.open(image_file))/255.,np.float32(Image.open(label_file))/255.      

        # sample = {'image': image, 'label': label}
        # if self.transform:
        #     sample = self.transform(sample)
        # sample['case_name'] = self.sample_list[idx].strip('\n')
        # return sample
