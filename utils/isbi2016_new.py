import os
import glob
import json
import torch
import random
import torch.nn as nn
import numpy as np
import torch.utils.data
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F
import cv2

import albumentations as A
from sklearn.model_selection import KFold
from .polar_transformations import centroid, to_polar


def norm01(x):
    return np.clip(x, 0, 255) / 255


seperable_indexes = json.load(open('utils/data_split.json', 'r'))


# cross validation
class myDataset(data.Dataset):
    def __init__(self, split, size=352, aug=False, polar=False):
        super(myDataset, self).__init__()
        self.polar = polar
        self.split = split

        # load images, label, point
        self.image_paths = []
        self.label_paths = []
        self.point_paths = []
        self.dist_paths = []

        root_dir = '/raid/wjc/data/skin_lesion/isic2016'

        if split == 'train':
            indexes = os.listdir(root_dir + '/Train/Image/')
            self.image_paths = [
                f'{root_dir}/Train/Image/{_id}' for _id in indexes
            ]
            self.label_paths = [
                f'{root_dir}/Train/Label/{_id[:-4]}_label.npy'
                for _id in indexes
            ]
            self.point_paths = [
                f'{root_dir}/Train/Point/{_id[:-4]}_label.npy'
                for _id in indexes
            ]

        elif split == 'valid':
            indexes = os.listdir(root_dir + '/Validation/Image/')
            self.image_paths = [
                f'{root_dir}/Validation/Image/{_id}' for _id in indexes
            ]
            self.label_paths = [
                f'{root_dir}/Validation/Label/{_id[:-4]}_label.npy'
                for _id in indexes
            ]
        else:
            indexes = os.listdir(root_dir + '/Test/Image/')
            self.image_paths = [
                f'{root_dir}/Test/Image/{_id}' for _id in indexes
            ]
            self.label_paths = [
                f'{root_dir}/Test/Label/{_id[:-4]}_label.npy'
                for _id in indexes
            ]

        print('Loaded {} frames'.format(len(self.image_paths)))
        self.num_samples = len(self.image_paths)
        self.aug = aug
        self.size = size

        p = 0.5
        self.transf = A.Compose([
            A.GaussNoise(p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.ShiftScaleRotate(p=p),
            #             A.RandomBrightnessContrast(p=p),
        ])

    def __getitem__(self, index):
        # print(self.image_paths[index])
        # image = cv2.imread(self.image_paths[index])
        # image_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # label_data = cv2.imread(self.label_paths[index], cv2.IMREAD_GRAYSCALE)
        image_data = np.load(self.image_paths[index])
        label_data = (np.load(self.label_paths[index]) * 255).astype('uint8')

        label_data = np.array(
            cv2.resize(label_data, (self.size, self.size), cv2.INTER_NEAREST))
        point_data = cv2.Canny(label_data, 0, 255) / 255.0 > 0.5
        label_data = label_data / 255. > 0.5
        image_data = np.array(
            cv2.resize(image_data, (self.size, self.size), cv2.INTER_LINEAR))
        if self.split == 'train':
            filter_point_data = (np.load(self.point_paths[index]) >
                                 0.7).astype('uint8')
            filter_point_data = np.array(
                cv2.resize(filter_point_data, (self.size, self.size),
                           cv2.INTER_NEAREST))
        else:
            filter_point_data = point_data.copy()

        # image_data = np.load(self.image_paths[index])
        # label_data = np.load(self.label_paths[index]) > 0.5
        # point_data = np.load(self.point_paths[index]) > 0.5
        # point_All_data = np.load(self.point_All_paths[index]) > 0.5  #

        #         label_data = np.expand_dims(label_data,-1)
        #         point_data = np.expand_dims(point_data,-1)
        if self.aug and self.split == 'train':
            mask = np.concatenate([
                label_data[..., np.newaxis].astype('uint8'),
                point_data[..., np.newaxis], filter_point_data[..., np.newaxis]
            ],
                                  axis=-1)
            #             print(mask.shape)
            tsf = self.transf(image=image_data.astype('uint8'), mask=mask)
            image_data, mask_aug = tsf['image'], tsf['mask']
            label_data = mask_aug[:, :, 0]
            point_data = mask_aug[:, :, 1]
            filter_point_data = mask_aug[:, :, 2]

        image_data = norm01(image_data)

        if self.polar:
            center = centroid(image_data)
            image_data = to_polar(image_data, center)
            label_data = to_polar(label_data, center) > 0.5

        label_data = np.expand_dims(label_data, 0)
        point_data = np.expand_dims(point_data, 0)
        filter_point_data = np.expand_dims(filter_point_data, 0)  #

        image_data = torch.from_numpy(image_data).float()
        label_data = torch.from_numpy(label_data).float()
        point_data = torch.from_numpy(point_data).float()
        filter_point_data = torch.from_numpy(filter_point_data).float()  #

        image_data = image_data.permute(2, 0, 1)
        return {
            'image_path': self.image_paths[index],
            'label_path': self.label_paths[index],
            # 'point_path': self.point_paths[index],
            'image': image_data,
            'label': label_data,
            'point': point_data,
            'filter_point_data': filter_point_data
        }

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    from tqdm import tqdm
    dataset = myDataset(split='train', aug=True)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=8,
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=True)
    import matplotlib.pyplot as plt
    for d in dataset:
        print(d['image'].shape, d['image'].max())
        print(d['point'].shape, d['point'].max())
        print(d['filter_point_data'].shape, d['filter_point_data'].max())
        image = d['image'].permute(1, 2, 0).cpu()
        point = d['point'][0].cpu()
        filter_point_data = d['filter_point_data'][0].cpu()
        plt.imshow(filter_point_data)
        plt.show()