import os, argparse, math
import numpy as np
from glob import glob
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from medpy.metric.binary import hd, dc, assd, jc

from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import time


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='xboundformer')
    parser.add_argument('--gpu', type=str, default='4')
    parser.add_argument('--net_layer', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='isic2016')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--fold', type=str)
    parser.add_argument('--lr_seg', type=float, default=1e-4)  #0.0003
    parser.add_argument('--n_epochs', type=int, default=150)  #100
    parser.add_argument('--bt_size', type=int, default=12)  #36
    parser.add_argument('--seg_loss', type=int, default=1, choices=[0, 1])
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--patience', type=int, default=500)  #50

    # transformer
    parser.add_argument('--filter', type=int, default=0)
    parser.add_argument('--im_num', type=int, default=1)
    parser.add_argument('--ex_num', type=int, default=1)
    parser.add_argument('--xbound', type=int, default=1)
    parser.add_argument('--point_w', type=float, default=1)

    #log_dir name
    parser.add_argument('--folder_name', type=str, default='Default_folder')

    parse_config = parser.parse_args()
    print(parse_config)
    return parse_config


def ce_loss(pred, gt):
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    return (-gt * torch.log(pred) - (1 - gt) * torch.log(1 - pred)).mean()


def structure_loss(pred, mask):
    """            TransFuse train loss        """
    """            Without sigmoid             """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


#-------------------------- train func --------------------------#
def train(epoch):
    model.train()
    iteration = 0
    for batch_idx, batch_data in enumerate(train_loader):
        #         print(epoch, batch_idx)
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        if parse_config.filter:
            point = (batch_data['filter_point_data'] > 0).cuda().float()
        else:
            point = (batch_data['point'] > 0).cuda().float()
        #point_All = (batch_data['point_All'] > 0).cuda().float()

        if parse_config.arch == 'transfuse':
            lateral_map_4, lateral_map_3, lateral_map_2 = model(data)

            loss4 = structure_loss(lateral_map_4, label)
            loss3 = structure_loss(lateral_map_3, label)
            loss2 = structure_loss(lateral_map_2, label)

            loss = 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 10 == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\t[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'
                    .format(epoch, batch_idx * len(data),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss2.item(),
                            loss3.item(), loss4.item()))

        else:
            P2, point_maps_pre, point_maps_pre1, point_maps_pre2 = model(data)
            if parse_config.im_num + parse_config.ex_num > 0:
                point_loss = 0.0
                point3 = F.max_pool2d(point, (32, 32), (32, 32))
                point2 = F.max_pool2d(point, (16, 16), (16, 16))
                point1 = F.max_pool2d(point, (8, 8), (8, 8))

                for point_pre, point_pre1, point_pre2 in zip(
                        point_maps_pre, point_maps_pre1, point_maps_pre2):
                    point_loss = point_loss + criteon(
                        point_pre, point1) + criteon(
                            point_pre1, point2) + criteon(point_pre2, point3)
                point_loss = point_loss / (3 * len(point_maps_pre1))
                seg_loss = 0.0
                for p in P2:
                    seg_loss = seg_loss + structure_loss(p, label)
                seg_loss = seg_loss / len(P2)
                loss = seg_loss + parse_config.point_w * point_loss

                if batch_idx % 50 == 0:
                    show_image = [label[0], F.sigmoid(P2[0][0])]
                    for point_map in [
                            point_maps_pre, point_maps_pre1, point_maps_pre2
                    ]:
                        tmp = F.interpolate(point_map[-1], size=(352, 352))[0]
                        show_image.append(tmp)
                    show_image = torch.cat(show_image, dim=2)
                    show_image = show_image.repeat(3, 1, 1)
                    show_image = torch.cat([data[0], show_image], dim=2)

                    writer.add_image('pred/all',
                                     show_image,
                                     epoch * len(train_loader) + batch_idx,
                                     dataformats='CHW')

            else:
                point_loss = 0.0
                seg_loss = 0.0
                for p in P2:
                    seg_loss = seg_loss + structure_loss(p, label)
                seg_loss = seg_loss / len(P2)
                loss = seg_loss + 2 * point_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 10 == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\t[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'
                    .format(epoch, batch_idx * len(data),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss,
                            seg_loss, point_loss))

    print("Iteration numbers: ", iteration)


#-------------------------- eval func --------------------------#
def evaluation(epoch, loader):
    model.eval()
    dice_value = 0
    iou_value = 0
    dice_average = 0
    iou_average = 0
    numm = 0
    for batch_idx, batch_data in enumerate(loader):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        point = (batch_data['point'] > 0).cuda().float()
        # point_All = (batch_data['point_data'] > 0).cuda().float()
        #point_All = nn.functional.max_pool2d(point_All,
        #                                 kernel_size=(16, 16),
        #                                 stride=(16, 16))

        with torch.no_grad():
            if parse_config.arch == 'transfuse':
                _, _, output = model(data)
                loss_fuse = structure_loss(output, label)
            elif parse_config.arch == 'xboundformer':
                output, point_maps_pre, point_maps_pre1, point_maps_pre2 = model(
                    data)
                loss = 0

            if parse_config.arch == 'transfuse':
                loss = loss_fuse

            output = output.cpu().numpy() > 0.5

        label = label.cpu().numpy()
        assert (output.shape == label.shape)
        dice_ave = dc(output, label)
        iou_ave = jc(output, label)
        dice_value += dice_ave
        iou_value += iou_ave
        numm += 1

    dice_average = dice_value / numm
    iou_average = iou_value / numm
    writer.add_scalar('val_metrics/val_dice', dice_average, epoch)
    writer.add_scalar('val_metrics/val_iou', iou_average, epoch)
    print("Average dice value of evaluation dataset = ", dice_average)
    print("Average iou value of evaluation dataset = ", iou_average)
    return dice_average, iou_average, loss


if __name__ == '__main__':
    #-------------------------- get args --------------------------#
    parse_config = get_cfg()

    #-------------------------- build loggers and savers --------------------------#
    exp_name = parse_config.dataset + '/' + parse_config.exp_name + '_loss_' + str(
        parse_config.seg_loss) + '_aug_' + str(
            parse_config.aug
        ) + '/' + parse_config.folder_name + '/fold_' + str(parse_config.fold)

    os.makedirs('logs/{}'.format(exp_name), exist_ok=True)
    os.makedirs('logs/{}/model'.format(exp_name), exist_ok=True)
    writer = SummaryWriter('logs/{}/log'.format(exp_name))
    save_path = 'logs/{}/model/best.pkl'.format(exp_name)
    latest_path = 'logs/{}/model/latest.pkl'.format(exp_name)

    EPOCHS = parse_config.n_epochs
    os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu
    device_ids = range(torch.cuda.device_count())

    #-------------------------- build dataloaders --------------------------#
    if parse_config.dataset == 'isic2018':
        from utils.isbi2018_new import norm01, myDataset

        dataset = myDataset(fold=parse_config.fold,
                            split='train',
                            aug=parse_config.aug)
        dataset2 = myDataset(fold=parse_config.fold, split='valid', aug=False)
    elif parse_config.dataset == 'isic2016':
        from utils.isbi2016_new import norm01, myDataset

        dataset = myDataset(split='train', aug=parse_config.aug)
        dataset2 = myDataset(split='valid', aug=False)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=parse_config.bt_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset2,
        batch_size=1,  #parse_config.bt_size
        shuffle=False,  #True
        num_workers=2,
        pin_memory=True,
        drop_last=False)  #True

    #-------------------------- build models --------------------------#
    if parse_config.arch is 'xboundformer':
        from lib.xboundformer import _segm_pvtv2
        model = _segm_pvtv2(1, parse_config.im_num, parse_config.ex_num,
                            parse_config.xbound, 352).cuda()
    elif parse_config.arch == 'transfuse':
        from lib.TransFuse.TransFuse import TransFuse_S
        model = TransFuse_S(pretrained=True).cuda()

    if len(device_ids) > 1:  # 多卡训练
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr_seg)

    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

    criteon = [None, ce_loss][parse_config.seg_loss]

    #-------------------------- start training --------------------------#

    max_dice = 0
    max_iou = 0
    best_ep = 0

    min_loss = 10
    min_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        start = time.time()
        train(epoch)
        dice, iou, loss = evaluation(epoch, val_loader)
        scheduler.step()

        if loss < min_loss:
            min_epoch = epoch
            min_loss = loss
        else:
            if epoch - min_epoch >= parse_config.patience:
                print('Early stopping!')
                break
        if iou > max_iou:
            max_iou = iou
            best_ep = epoch
            torch.save(model.state_dict(), save_path)
        else:
            if epoch - best_ep >= parse_config.patience:
                print('Early stopping!')
                break
        torch.save(model.state_dict(), latest_path)
        time_elapsed = time.time() - start
        print(
            'Training and evaluating on epoch:{} complete in {:.0f}m {:.0f}s'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))
