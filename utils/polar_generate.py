import cv2
import os
import random
import torch
import numpy as np
import skimage.draw
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from polar_transformations import to_polar, centroid


def polar_gen_isic2018():
    data_dir = '/raid/wjc/data/skin_lesion/isic2018_jpg_smooth/'

    os.makedirs(data_dir + '/PolarImage', exist_ok=True)
    os.makedirs(data_dir + '/PolarLabel', exist_ok=True)

    path_list = os.listdir(data_dir + '/Label/')
    path_list.sort()
    num = 0
    for path in tqdm(path_list):
        image_data = cv2.imread(os.path.join(data_dir, 'Image', path))

        label_data = cv2.imread(os.path.join(data_dir, 'Label', path),
                                cv2.IMREAD_GRAYSCALE)
        center = centroid(image_data)
        image_data = to_polar(image_data, center)
        label_data = to_polar(label_data, center)
        # print(image_data.max(), label_data.max())

        cv2.imwrite(data_dir + '/PolarImage/' + path, image_data)
        cv2.imwrite(data_dir + '/PolarLabel/' + path, label_data)
        # break


def point_gen_isic2016():
    R = 10
    N = 25
    for split in ['Train', 'Test', 'Validation']:
        data_dir = '/raid/wjc/data/skin_lesion/isic2016/{}/Label'.format(split)

        save_dir = data_dir.replace('Label', 'Point')
        os.makedirs(save_dir, exist_ok=True)

        path_list = os.listdir(data_dir)
        path_list.sort()
        num = 0
        for path in tqdm(path_list):
            name = path[:-4]
            label_path = os.path.join(data_dir, path)
            print(label_path)
            label_ori, point_heatmap = kpm_gen(label_path, R, N)
            save_path = os.path.join(save_dir, name + '.npy')
            np.save(save_path, point_heatmap)
            num += 1


if __name__ == '__main__':
    polar_gen_isic2018()