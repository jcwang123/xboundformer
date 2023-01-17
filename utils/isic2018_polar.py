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
from utils.polar_transformations import centroid, to_polar


class isic2018Dataset(data.Dataset):
    """
    dataloader for isic2018 segmentation tasks
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

            self.transform = A.Compose([ToTensorV2()])
        else:
            print('no augmentation')
            self.transform = A.Compose([
                # A.Resize(self.trainsize, self.trainsize),
                ToTensorV2()
            ])

    def __getitem__(self, idx):
        file_name = self.images[idx]
        # gt_name = file_name[:-4] + '_segmentation.png'
        img_root = os.path.join(self.image_root, file_name)
        gt_root = os.path.join(self.gt_root, file_name)
        image = cv2.imread(img_root)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = (cv2.imread(gt_root, cv2.IMREAD_GRAYSCALE))
        gt = gt // 255.0

        # if self.polar:
        #     if self.manual_centers is not None:
        #     center = self.manual_centers[idx]
        # else:
        #     center = polar_transformations.centroid(label)
        center = centroid(gt)

        image = to_polar(image, center)
        gt = to_polar(gt, center)

        gt = np.concatenate([gt[..., np.newaxis], gt[..., np.newaxis]],
                            axis=-1)
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


# def get_loader(image_root,batchsize, trainsize, floder,shuffle=True, num_workers=8, pin_memory=True, augmentation=False):
#     train_image_index_all,test_image_index_all=create_k_fold_division(image_root)
#     image_index=train_image_index_all[floder]
#     test_index=test_image_index_all[floder]
#     dataset = isic2018Dataset(image_root, image_index, trainsize, augmentation)
#     data_loader = data.DataLoader(dataset=dataset,
#                                   batch_size=batchsize,
#                                   shuffle=shuffle,
#                                   num_workers=num_workers,
#                                   pin_memory=pin_memory)

#     testset=test_dataset(image_root,test_index,trainsize)
#     test_loader = data.DataLoader(dataset=testset,
#                                   batch_size=1,
#                                   shuffle=shuffle,
#                                   num_workers=num_workers,
#                                   pin_memory=pin_memory)
#     return data_loader,testset


def get_loader(image_root,
               gt_root,
               batchsize,
               trainsize,
               floder,
               shuffle=True,
               num_workers=8,
               pin_memory=True,
               augmentation=False):
    js = json.load(open('utils/data_split.json'))
    # print(js)
    # train / test
    all_index = [f for f in os.listdir(image_root) if f.endswith('.jpg')]

    test_index = ['ISIC_' + i + '.jpg' for i in js[str(floder)]]
    image_index = list(filter(lambda x: x not in test_index, all_index))
    print(len(all_index), len(image_index), len(test_index))

    dataset = isic2018Dataset(image_root, gt_root, image_index, trainsize,
                              augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    testset = isic2018Dataset(image_root, gt_root, test_index, trainsize,
                              False)
    test_loader = data.DataLoader(dataset=testset,
                                  batch_size=1,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader, test_loader


class test_dataset:
    def __init__(self, image_root, gt_root, test_index, testsize):
        self.testsize = testsize
        self.image_root = image_root
        self.gt_root = gt_root
        self.images = test_index
        self.transform = A.Compose(
            [A.Resize(self.testsize, self.testsize),
             ToTensorV2()])
        self.size = len(self.images)
        self.index = 0

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.image_root, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # gt_ind=self.images[idx].split('_')
        # gt_name=gt_ind[0]+'_Task1_'+gt_ind[2]+'_GroundTruth/ISIC_'+gt_ind[4][:-4]+'_segmentation.png'
        gt_name = self.images[idx][:-4] + '_segmentation.png'
        gt_root = os.path.join(self.gt_root, gt_name)
        gt = cv2.imread(gt_root, cv2.IMREAD_GRAYSCALE)
        pair = self.transform(image=image, mask=gt)
        name = self.images[idx].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        image = pair['image'].unsqueeze(0) / 255
        gt = pair['mask'] / 255
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


if __name__ == '__main__':
    isic2018Dataset()