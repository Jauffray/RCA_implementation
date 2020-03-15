from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from . import paired_transforms_tv04 as p_tr

import os.path as osp
import pandas as pd
from PIL import Image
import numpy as np
from skimage.measure import regionprops

import torch
import torchvision.transforms.functional as F


class DRIVE_from_csv(Dataset):
    def __init__(self, csv_path, in_channels=3, transforms=None, normalize='from_im_max', label_values=None):
        df = pd.read_csv(csv_path)
        self.im_list = df.im_paths
        self.gt_list = df.gt_paths
        self.mask_list = df.mask_paths

        self.in_channels = in_channels
        self.transforms = transforms
        self.normalize = normalize
        self.label_values = label_values  # for use in label_encoding

    def label_encoding(self, gdt):
        gdt_gray = np.array(gdt.convert('L'))
        classes = np.arange(len(self.label_values))
        for i in classes:
            gdt_gray[gdt_gray == self.label_values[i]] = classes[i]
        return Image.fromarray(gdt_gray)

    def crop_to_fov(self, img, target, mask):
        minr, minc, maxr, maxc = regionprops(np.array(mask))[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        tg_crop = Image.fromarray(np.array(target)[minr:maxr, minc:maxc])
        mask_crop = Image.fromarray(np.array(mask)[minr:maxr, minc:maxc])
        return im_crop, tg_crop, mask_crop

    # def normalizer(self, torch_tensor):
    #     '''
    #     This will return the tensor after subtracting mean and dividing
    #     by std (channel-wise). To be called after torchvision.transforms.ToTensor(),
    #     because it expects CxHxW
    #     '''
        # check if std == 0 (-> then, image is constant, and Normalization will fail).
        # if so, simply return a tensor of zeros
        # if (torch_tensor.std(axis=[1, 2]) > 0).any().item() is False: #error in recent pytorch
        #     print('Be careful, this is a constant image')
        #     return torch.zeros_like(torch_tensor)
        # else:
        #     return F.normalize(torch_tensor, torch_tensor.mean(axis=[1, 2]), torch_tensor.std(axis=[1, 2]))

    def __getitem__(self, index):
        # load image and labels
        img = Image.open(self.im_list[index])
        if self.in_channels == 1:
            _, img, _ = img.split()
        target = Image.open(self.gt_list[index])
        mask = Image.open(self.mask_list[index]).convert('L')

        img, target, mask = self.crop_to_fov(img, target, mask)

        target = self.label_encoding(target)

        target = np.array(self.label_encoding(target))
        target[np.array(mask) == 0] = 0
        target = Image.fromarray(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.normalize!='from_im_max':
            raise NotImplementedError
            # return self.normalizer(img), target

        return img, target

    def __len__(self):
        return len(self.im_list)

class TestDataset(Dataset):
    def __init__(self, csv_path, tg_size, in_channels=3, normalize='from_im_max'):
        df = pd.read_csv(csv_path)
        self.im_list = df.im_paths
        self.mask_list = df.mask_paths
        self.in_channels = in_channels
        self.normalize = normalize
        self.tg_size = tg_size

    def crop_to_fov(self, img, mask):
        mask = np.array(mask).astype(int)
        minr, minc, maxr, maxc = regionprops(mask)[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        return im_crop, [minr, minc, maxr, maxc]

    def normalizer(self, torch_tensor):
        '''
        This will return the tensor after subtracting mean and dividing
        by std (channel-wise). To be called after torchvision.transforms.ToTensor(),
        because it expects CxHxW
        '''
        # check if std == 0 (-> then, image is constant, and Normalization will fail).
        # if so, simply return a tensor of zeros
        if (torch_tensor.std(axis=[1, 2]) > 0).any().item() is False:
            print('Be careful, this is a constant image')
            return torch.zeros_like(torch_tensor)
        else:
            return F.normalize(torch_tensor, torch_tensor.mean(axis=[1, 2]), torch_tensor.std(axis=[1, 2]))

    def __getitem__(self, index):
        # load image and mask
        img = Image.open(self.im_list[index])
        if self.in_channels == 1:
            _, img, _ = img.split()
        mask = Image.open(self.mask_list[index]).convert('L')
        img, coords_crop = self.crop_to_fov(img, mask)
        original_sz = img.size[1], img.size[0]  # in numpy convention

        rsz = p_tr.Resize(self.tg_size)
        tnsr = p_tr.ToTensor()
        tr = p_tr.Compose([rsz, tnsr])
        img = tr(img)  # only transform image

        if self.normalize == 'from_im': raise NotImplementedError
            # return self.normalizer(img), np.array(mask).astype(bool), coords_crop, original_sz, self.im_list[index]
        elif self.normalize == 'from_dataset': raise NotImplementedError
            # return self.normalizer(img), np.array(mask).astype(bool), coords_crop, original_sz, self.im_list[index]
        elif self.normalize == 'from_im_max':
            pass  # this is what ToTensor() does
        else:
            raise NotImplementedError
        return img, np.array(mask).astype(bool), coords_crop, original_sz, self.im_list[index]

    def __len__(self):
        return len(self.im_list)

def get_train_val_datasets(path_data, in_channels=3, normalize='from_im_max', aug='all', solo=False):
    if(solo):
        path_train_csv = osp.join(path_data, 'solo.csv')
        path_val_csv = osp.join(path_data, 'solo.csv')
    else:
        path_train_csv = osp.join(path_data, 'train.csv')
        path_val_csv = osp.join(path_data, 'val.csv')

    train_dataset = DRIVE_from_csv(csv_path=path_train_csv, in_channels=in_channels,
                                   normalize=normalize, label_values=[0, 255])
    val_dataset = DRIVE_from_csv(csv_path=path_val_csv, in_channels=in_channels,
                                 normalize=normalize, label_values=[0, 255])
    # transforms definition
    size = 512, 512
    # required transforms
    resize = p_tr.Resize(size)
    tensorizer = p_tr.ToTensor()
    # geometric transforms
    h_flip = p_tr.RandomHorizontalFlip()
    v_flip = p_tr.RandomVerticalFlip()
    rotate = p_tr.RandomRotation(degrees=180)
    scale = p_tr.RandomAffine(degrees=0, scale=(0.85, 1.15))
    transl = p_tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = p_tr.RandomChoice([scale, transl, rotate])
    # intensity transforms
    brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
    jitter = p_tr.ColorJitter(brightness, contrast, saturation, hue)

    val_transforms = p_tr.Compose([resize, tensorizer])
    if aug == 'no':
        train_transforms = p_tr.Compose([resize, tensorizer])
    elif aug == 'geom':
        train_transforms = p_tr.Compose([resize, scale_transl_rot, h_flip, v_flip,tensorizer])
    elif aug == 'all':
        train_transforms = p_tr.Compose([resize, jitter, scale_transl_rot, h_flip, v_flip, tensorizer])
    else:
        raise Exception('Not a supported set of transforms')

    if normalize == 'from_im': raise NotImplementedError
        # train_transforms.insert(-2, p_tr.Normalize(mean=0, std=1))
        # val_transforms.insert(-2, p_tr.Normalize(mean=0, std=1))
    elif normalize == 'from_dataset': raise NotImplementedError
        # train_transforms.insert(-2, p_tr.Normalize(mean=0, std=1))
        # val_transforms.insert(-2, p_tr.Normalize(mean=0, std=1))
    elif normalize == 'from_im_max': pass # this is what ToTensor() does
    else: raise NotImplementedError

    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms

    return train_dataset, val_dataset

def get_train_val_loaders(path_data, in_channels=3, normalize=False, batch_size=4, aug='all', solo=False):
    train_dataset, val_dataset = get_train_val_datasets(path_data, in_channels=in_channels,
                                                        normalize='from_im_max', aug=aug, solo=solo)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=0)

    return train_loader, val_loader

def get_test_dataset(data_path, csv_path='test.csv', tg_size=(512, 512),
                     in_channels=3, normalize='from_im_max'):
    # csv_path will only not be test.csv when we want to build drive predictions
    path_test_csv = osp.join(data_path, csv_path)
    test_dataset = TestDataset(csv_path=path_test_csv, in_channels=in_channels, normalize=normalize, tg_size=tg_size)

    return test_dataset
