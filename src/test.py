from medpy.metric.binary import hd, hd95, dc, jc, assd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
import sys
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from lib.xboundformer import _segm_pvtv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_point_pred = True


def isbi2016():
    target_size = (512, 512)

    for model_name in ['xboundformer']:
        if model_name == 'xboundformer':
            model = _segm_pvtv2(1, 2, 2, 1, 352).to(device)
        else:
            # TODO
            raise NotImplementedError
        model.load_state_dict(
            torch.load(
                f'logs/isbi2016/test_loss_1_aug_1/{model_name}/fold_None/model/best.pkl'
            ))
        for fold in ['PH2', 'Test']:
            save_dir = f'results/ISIC-2016-pictures/{model_name}/{fold}'
            os.makedirs(save_dir, exist_ok=True)
            from utils.isbi2016_new import norm01, myDataset
            if fold == 'PH2':
                dataset = myDataset(split='test', aug=False)
            else:
                dataset = myDataset(split='valid', aug=False)
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

            model.eval()
            for batch_idx, batch_data in tqdm(enumerate(test_loader)):
                data = batch_data['image'].to(device).float()
                label = batch_data['label'].to(device).float()
                path = batch_data['image_path'][0]
                with torch.no_grad():
                    output, point_pred1, point_pred2, point_pred3 = model(data)
                if save_point_pred:
                    os.makedirs(save_dir.replace('pictures', 'point_maps'),
                                exist_ok=True)
                    point_pred1 = F.interpolate(point_pred1[-1], target_size)
                    point_pred1 = point_pred1.cpu().numpy()[0, 0]
                    plt.imsave(
                        save_dir.replace('pictures', 'point_maps') + '/' +
                        os.path.basename(path)[:-4] + '.png', point_pred1)
                output = torch.sigmoid(output)[0][0]
                output = (output.cpu().numpy() > 0.5).astype('uint8')
                output = (cv2.resize(output, target_size, cv2.INTER_NEAREST) >
                          0.5) * 1
                plt.imsave(
                    save_dir + '/' + os.path.basename(path)[:-4] + '.png',
                    output)


def isbi2018():
    model = _segm_pvtv2(1, 1, 1, 1, 352).to(device)
    target_size = (512, 512)
    for fold in range(5):
        model.load_state_dict(
            torch.load(
                f'logs/isbi2018/test_loss_1_aug_1/xboundformer/fold_{fold}/model/best.pkl'
            ))
        save_dir = f'results/ISIC-2018-pictures/xboundformer/fold-{int(fold)+1}'
        os.makedirs(save_dir, exist_ok=True)
        from utils.isbi2018_new import norm01, myDataset
        dataset = myDataset(fold=str(fold), split='valid', aug=False)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        model.eval()
        for batch_idx, batch_data in tqdm(enumerate(test_loader)):
            data = batch_data['image'].to(device).float()
            label = batch_data['label'].to(device).float()
            path = batch_data['image_path'][0]
            with torch.no_grad():
                output, _, _, _ = model(data)
            output = torch.sigmoid(output)[0][0]
            output = (output.cpu().numpy() > 0.5).astype('uint8')
            output = (cv2.resize(output, target_size, cv2.INTER_NEAREST) >
                      0.5) * 1
            plt.imsave(
                save_dir + '/' + os.path.basename(path).split('_')[1][:-4] +
                '.png', output)


def isbi2018_ablation(folder_name):
    vs = list(map(int, folder_name.split('_')[1:]))
    model = _segm_pvtv2(1, vs[0], vs[1], vs[2], 352).to(device)
    target_size = (512, 512)
    for fold in range(5):
        model.load_state_dict(
            torch.load(
                f'logs/isbi2018/test_loss_1_aug_1/{folder_name}/fold_{fold}/model/best.pkl'
            ))
        save_dir = f'results/ISIC-2018-pictures/{folder_name}/fold-{int(fold)+1}'
        os.makedirs(save_dir, exist_ok=True)
        from utils.isbi2018_new import norm01, myDataset
        dataset = myDataset(fold=str(fold), split='valid', aug=False)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        model.eval()
        for batch_idx, batch_data in tqdm(enumerate(test_loader)):
            data = batch_data['image'].to(device).float()
            label = batch_data['label'].to(device).float()
            path = batch_data['image_path'][0]
            with torch.no_grad():
                output, _, _, _ = model(data)
            output = torch.sigmoid(output)[0][0]
            output = (output.cpu().numpy() > 0.5).astype('uint8')
            output = (cv2.resize(output, target_size, cv2.INTER_NEAREST) >
                      0.5) * 1
            plt.imsave(
                save_dir + '/' + os.path.basename(path).split('_')[1][:-4] +
                '.png', output)


if __name__ == '__main__':
    # isbi2016()
    isbi2018_ablation('bl_0_0_0')
    isbi2018_ablation('bl_1_0_0')
    isbi2018_ablation('bl_1_1_0')
    isbi2018_ablation('bl_1_1_1')
