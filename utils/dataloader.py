import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        self.images = [
            image_root + f for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ]
        self.gts = [
            gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')
        ]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.color1, self.color2 = [], []
        for name in self.images:
            if os.path.basename(name)[:-4].isdigit():
                self.color1.append(name)
            else:
                self.color2.append(name)
        if self.augmentations:
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
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        name2 = self.color1[idx % len(self.color1)] if np.random.rand(
        ) < 0.7 else self.color2[idx % len(self.color2)]
        image2 = cv2.imread(name2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

        mean, std = image.mean(axis=(0, 1),
                               keepdims=True), image.std(axis=(0, 1),
                                                         keepdims=True)
        mean2, std2 = image2.mean(axis=(0, 1),
                                  keepdims=True), image2.std(axis=(0, 1),
                                                             keepdims=True)
        image = np.uint8((image - mean) / std * std2 + mean2)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        gt = (cv2.imread(self.gts[idx], cv2.IMREAD_GRAYSCALE))
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

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

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


def get_loader(image_root,
               gt_root,
               batchsize,
               trainsize,
               shuffle=True,
               num_workers=4,
               pin_memory=True,
               augmentation=False):

    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [
            image_root + f for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ]
        self.gts = [
            gt_root + f for f in os.listdir(gt_root)
            if f.endswith('.tif') or f.endswith('.png')
        ]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = A.Compose(
            [A.Resize(self.testsize, self.testsize),
             ToTensorV2()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = cv2.imread(self.images[self.index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.gts[self.index], cv2.IMREAD_GRAYSCALE)
        pair = self.transform(image=image, mask=gt)
        image = pair['image'].unsqueeze(0) / 255
        gt = pair['mask'] / 255
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')