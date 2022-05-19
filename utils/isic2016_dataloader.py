import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json


class isic2016Dataset(data.Dataset):
    """
    dataloader for isic2016 segmentation tasks
    """
    def __init__(self, image_root, gt_root, image_index, trainsize,
                 augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        self.image_root = image_root
        self.gt_root = gt_root
        self.images = image_index
        self.size = len(self.images)

        if self.augmentations:
            print('Using RandomRotation, RandomFlip')

            self.transform = A.Compose([
                A.Rotate(90),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Resize(self.trainsize, self.trainsize),
                ToTensorV2()
            ])
        else:
            print('no augmentation')
            self.transform = A.Compose(
                [A.Resize(self.trainsize, self.trainsize),
                 ToTensorV2()])

    def __getitem__(self, idx):
        file_name = self.images[idx]
        # gt_name = file_name[:-4] + '_segmentation.png'
        img_root = os.path.join(self.image_root, file_name)
        gt_root = os.path.join(self.gt_root, file_name[:-4] + '_label.npy')
        # image = cv2.imread(img_root)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # gt = (cv2.imread(gt_root, cv2.IMREAD_GRAYSCALE))
        image = np.load(img_root)
        gt = (np.load(gt_root) * 255).astype(np.uint8)

        point_heatmap = cv2.Canny(gt, 0, 255) / 255.0
        gt = gt // 255.0
        gt = np.concatenate(
            [gt[..., np.newaxis], point_heatmap[..., np.newaxis]], axis=-1)
        pair = self.transform(image=image, mask=gt)
        gt = pair['mask'][:, :, 0]
        point_heatmap = pair['mask'][:, :, 1]
        gt = torch.unsqueeze(gt, 0)
        point_heatmap = torch.unsqueeze(point_heatmap, 0)
        image = pair['image'] / 255.0

        return image, gt, point_heatmap

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h),
                                                                 Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(root_path,
               batchsize,
               trainsize,
               shuffle=True,
               num_workers=8,
               pin_memory=True,
               augmentation=False):

    dataset = isic2016Dataset(root_path + 'Train/Image',
                              root_path + 'Train/Label',
                              os.listdir(root_path + 'Train/Image'), trainsize,
                              augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    validset = isic2016Dataset(root_path + 'Validation/Image',
                               root_path + 'Validation/Label',
                               os.listdir(root_path + 'Validation/Image'),
                               trainsize, False)
    valid_loader = data.DataLoader(dataset=validset,
                                   batch_size=1,
                                   shuffle=shuffle,
                                   num_workers=num_workers,
                                   pin_memory=pin_memory)

    testset = isic2016Dataset(root_path + 'Test/Image',
                              root_path + 'Test/Label',
                              os.listdir(root_path + 'Test/Image'), trainsize,
                              False)
    test_loader = data.DataLoader(dataset=testset,
                                  batch_size=1,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader, valid_loader, test_loader