import os
import random
import numpy as np
import torch
import SimpleITK as sitk
import nibabel as nib
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(self.split+'.txt').readlines()
        self.data_dir = base_dir
        
        self.img_ids = [i_id.strip() for i_id in self.sample_list]
        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(list_dir, "data\PANCREAS_%s.nii" % name)
            label_file = os.path.join(list_dir, "label\label%s.nii" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        datafiles = self.files[idx]
        '''load the datas'''
        # name = datafiles["name"]
        
        # image_nii_file = sitk.ReadImage(datafiles["img"])
        # image_nii = sitk.GetArrayFromImage(image_nii_file)
        # label_nii_file = sitk.ReadImage(datafiles["label"])
        # label_nii = sitk.GetArrayFromImage(label_nii_file)
        # del image_nii_file,label_nii_file
        image_nii = nib.load(datafiles["img"]).dataobj
        label_nii = nib.load(datafiles["label"]).dataobj

        _,_,lenth=image_nii.shape
        
        right_num=random.sample(range(0,lenth),1)[0] # 随机一个slice开始遍历找到非零label
        for i in range(right_num,lenth):
            if ~np.all(label_nii[:,:,i]==0):
                break
        if np.all(label_nii[:,:,i]==0):
            for i in range(0,right_num):
                if ~np.all(label_nii[:,:,i]==0):
                    break
        

        image = image_nii[:,:,i]/1.0

        label = label_nii[:,:,i]/1.0
        

        return np.float32(image),np.float32(label)      

        # sample = {'image': image, 'label': label}
        # if self.transform:
        #     sample = self.transform(sample)
        # sample['case_name'] = self.sample_list[idx].strip('\n')
        # return sample
