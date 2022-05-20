from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.modules import xboundlearner, xboundlearnerv2, _simple_learner
from lib.vision_transformers import in_scale_transformer

from lib.pvtv2 import pvt_v2_b2  #


def _segm_pvtv2(num_classes, im_num, ex_num, xbound, trainsize):
    backbone = pvt_v2_b2(img_size=trainsize)

    if 1:
        path = 'pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = backbone.state_dict()
        state_dict = {
            k: v
            for k, v in save_model.items() if k in model_dict.keys()
        }
        model_dict.update(state_dict)
        backbone.load_state_dict(model_dict)
    classifier = _simple_classifier(num_classes)
    model = _SimpleSegmentationModel(backbone, classifier, im_num, ex_num,
                                     xbound)
    return model


class _simple_classifier(nn.Module):
    def __init__(self, num_classes):
        super(_simple_classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(192, 64, 1, padding=1, bias=False),  #560
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1))
        self.classifier1 = nn.Sequential(nn.Conv2d(128, num_classes, 1))
        self.classifier2 = nn.Sequential(nn.Conv2d(128, num_classes, 1))
        self.classifier3 = nn.Sequential(nn.Conv2d(128, num_classes, 1))

    def forward(self, feature):
        low_level_feature = feature[0]
        output_feature = feature[1]
        output_feature = F.interpolate(output_feature,
                                       size=low_level_feature.shape[2:],
                                       mode='bilinear',
                                       align_corners=False)
        if self.training:
            return [
                self.classifier(
                    torch.cat([low_level_feature, output_feature], dim=1)),
                self.classifier1(feature[1]),
                self.classifier2(feature[2]),
                self.classifier3(feature[3])
            ]
        else:
            return self.classifier(
                torch.cat([low_level_feature, output_feature], dim=1))


class _SimpleSegmentationModel(nn.Module):
    # general segmentation model
    def __init__(self, backbone, classifier, im_num, ex_num, xbound):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.bat_low = _bound_learner(hidden_features=128,
                                      im_num=im_num,
                                      ex_num=ex_num,
                                      xbound=xbound)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(
            x
        )  # ([8, 64, 64, 64]) ([8, 128, 32, 32]) ([8, 320, 16, 16]) ([8, 512, 8, 8])
        features, point_pre1, point_pre2, point_pre3 = self.bat_low(features)
        outputs = self.classifier(features)
        if self.training:
            outputs = [
                F.interpolate(o,
                              size=input_shape,
                              mode='bilinear',
                              align_corners=False) for o in outputs
            ]
        else:
            outputs = F.interpolate(outputs,
                                    size=input_shape,
                                    mode='bilinear',
                                    align_corners=False)
        return outputs, point_pre1, point_pre2, point_pre3


class _bound_learner(nn.Module):
    def __init__(self,
                 point_pred=1,
                 hidden_features=128,
                 im_num=2,
                 ex_num=2,
                 xbound=True):

        super().__init__()
        self.im_num = im_num
        self.ex_num = ex_num

        self.point_pred = point_pred

        self.convolution_mapping_1 = nn.Conv2d(in_channels=128,
                                               out_channels=hidden_features,
                                               kernel_size=(1, 1),
                                               stride=(1, 1),
                                               padding=(0, 0),
                                               bias=True)
        self.convolution_mapping_2 = nn.Conv2d(in_channels=320,
                                               out_channels=hidden_features,
                                               kernel_size=(1, 1),
                                               stride=(1, 1),
                                               padding=(0, 0),
                                               bias=True)
        self.convolution_mapping_3 = nn.Conv2d(in_channels=512,
                                               out_channels=hidden_features,
                                               kernel_size=(1, 1),
                                               stride=(1, 1),
                                               padding=(0, 0),
                                               bias=True)
        normalize_before = True

        if im_num + ex_num > 0:
            self.im_ex_boud1 = in_scale_transformer(
                point_pred_layers=1,
                num_encoder_layers=im_num,
                num_decoder_layers=ex_num,
                d_model=hidden_features,
                nhead=8,
                normalize_before=normalize_before)
            self.im_ex_boud2 = in_scale_transformer(
                point_pred_layers=1,
                num_encoder_layers=im_num,
                num_decoder_layers=ex_num,
                d_model=hidden_features,
                nhead=8,
                normalize_before=normalize_before)
            self.im_ex_boud3 = in_scale_transformer(
                point_pred_layers=1,
                num_encoder_layers=im_num,
                num_decoder_layers=ex_num,
                d_model=hidden_features,
                nhead=8,
                normalize_before=normalize_before)
            # self.cross_attention_3_1 = xboundlearner(hidden_features, 8)
            # self.cross_attention_3_2 = xboundlearner(hidden_features, 8)

        self.xbound = xbound
        if xbound:
            self.cross_attention_3_1 = xboundlearnerv2(hidden_features, 8)
            self.cross_attention_3_2 = xboundlearnerv2(hidden_features, 8)
        else:
            self.cross_attention_3_1 = _simple_learner(hidden_features)
            self.cross_attention_3_2 = _simple_learner(hidden_features)

        self.trans_out_conv = nn.Conv2d(hidden_features * 2, 512, 1, 1)  #

    def forward(self, x):
        # for tmp in x:
        #     print(tmp.size())
        features_1 = x[1]
        features_2 = x[2]
        features_3 = x[3]
        features_1 = self.convolution_mapping_1(features_1)
        features_2 = self.convolution_mapping_2(features_2)
        features_3 = self.convolution_mapping_3(features_3)

        # in-scale attention
        if self.im_num + self.ex_num > 0:
            latent_tensor_1, features_encoded_1, point_maps_1 = self.im_ex_boud1(
                features_1)

            latent_tensor_2, features_encoded_2, point_maps_2 = self.im_ex_boud2(
                features_2)

            latent_tensor_3, features_encoded_3, point_maps_3 = self.im_ex_boud3(
                features_3)

            # cross-scale attention6
            if self.ex_num > 0:
                latent_tensor_1 = latent_tensor_1.permute(2, 0, 1)
                latent_tensor_2 = latent_tensor_2.permute(2, 0, 1)
                latent_tensor_3 = latent_tensor_3.permute(2, 0, 1)

        else:
            features_encoded_1 = features_1
            features_encoded_2 = features_2
            features_encoded_3 = features_3

        # ''' point map Upsample '''
        if self.xbound:
            features_encoded_2_2 = self.cross_attention_3_2(
                features_encoded_2, features_encoded_3, latent_tensor_2,
                latent_tensor_3)
            features_encoded_1_2 = self.cross_attention_3_1(
                features_encoded_1, features_encoded_2_2, latent_tensor_1,
                latent_tensor_2)
        else:
            features_encoded_2_2 = self.cross_attention_3_2(
                features_encoded_2, features_encoded_3)
            features_encoded_1_2 = self.cross_attention_3_1(
                features_encoded_1, features_encoded_2_2)

        # trans_feature_maps = self.trans_out_conv(
        #     torch.cat([features_encoded_3_1, features_encoded_3_2], dim=1))

        # x[3] = trans_feature_maps
        # x[2] = torch.cat([x[2], features_encoded_2], dim=1)
        # x[1] = torch.cat([x[1], features_encoded_1], dim=1)

        features_stage2 = [
            x[0], features_encoded_1_2, features_encoded_2_2,
            features_encoded_3
        ]

        if self.im_num + self.ex_num > 0:
            return features_stage2, point_maps_1, point_maps_2, point_maps_3  #
        else:
            return features_stage2, None, None, None  #


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    model = _segm_pvtv2(1).cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1 = model(input_tensor)
