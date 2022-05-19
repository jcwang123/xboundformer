from lib.pvtv2 import pvt_v2_b2  #
import torch
from torch import nn
import torch.nn.functional as F
# from lib.Vision_Transformer import detr_Transformer, detr_BA_Transformer
from lib.modules import BoundaryCrossAttention
from lib.transformer import BoundaryAwareTransformer
from lib.replknet import RepLKBlock


def _segm_pvtv2(name, backbone_name, num_classes, output_stride,
                pretrained_backbone):

    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = pvt_v2_b2()
    if pretrained_backbone:
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = backbone.state_dict()
        state_dict = {
            k: v
            for k, v in save_model.items() if k in model_dict.keys()
        }
        model_dict.update(state_dict)
        backbone.load_state_dict(model_dict)

    inplanes = 512
    low_level_planes = 64

    if name == 'deeplabv3plus':
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes,
                                       aspp_dilate)

    model = DeepLabV3(backbone, classifier)
    return model


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self,
                 in_channels,
                 low_level_channels,
                 num_classes,
                 aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(560, 256, 3, padding=1, bias=False),  #
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1))
        self._init_weight()

    def forward(self, feature):
        # low_level_feature = self.project( feature['low_level'] )
        # output_feature = self.aspp(feature['out'])
        low_level_feature = self.project(feature[0])
        output_feature = feature[3]
        # output_feature = self.aspp(feature[3])
        output_feature = F.interpolate(output_feature,
                                       size=low_level_feature.shape[2:],
                                       mode='bilinear',
                                       align_corners=False)
        return self.classifier(
            torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        # self.bat=BAT(num_classes=1,point_pred=1, in_channels=512,decoder=True, transformer_type_index=0)
        # self.bat_low=BAT(num_classes=1,point_pred=1,in_channels=320,hidden_features=320, decoder=False, transformer_type_index=0)
        # self.bat=BAT(in_channel=512)
        self.bat_low = BAT(in_channel=320)
        # self.bat0=BAT(in_channel=64)
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(
            x
        )  # ([8, 64, 64, 64]) ([8, 128, 32, 32]) ([8, 320, 16, 16]) ([8, 512, 8, 8])
        # features[3],point_pre=self.bat(features[3])
        features[2], point_pre = self.bat_low(features[2])
        # features[0],point_pre=self.bat0(features[0])
        x = self.classifier(features)
        x = F.interpolate(x,
                          size=input_shape,
                          mode='bilinear',
                          align_corners=False)
        # p=F.interpolate(point_pre, size=input_shape, mode='bilinear', align_corners=False)
        return x, point_pre


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels,
                      out_channels,
                      3,
                      padding=dilation,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(11),  ###
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x,
                             size=size,
                             mode='bilinear',
                             align_corners=False)


def deeplabv3plus_pvtv2(num_classes=1,
                        output_stride=8,
                        pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a pvtv2 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _segm_pvtv2('deeplabv3plus',
                       'pvtv2',
                       num_classes,
                       output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


class BAT(nn.Module):
    def __init__(self, in_channel=512):
        super(BAT, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=512,
                      kernel_size=(3, 3),
                      padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=256,
                      kernel_size=(3, 3),
                      padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=self.in_channel,
                      kernel_size=(3, 3),
                      padding=1), nn.BatchNorm2d(self.in_channel), nn.ReLU())
        # self.conv1=RepLKBlock(self.in_channel,128,31,5,0.)
        # self.conv2=RepLKBlock(self.in_channel,256,29,5,0.1)
        # self.conv3=RepLKBlock(self.in_channel,512,27,5,0.2)
        # self.conv5=RepLKBlock(self.in_channel,1024,13,5,0.3)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=1,
                      kernel_size=(1, 1)), nn.BatchNorm2d(1), nn.ReLU())
        # self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        point1 = self.conv1(x)
        # # point1=point1+x
        point1 = self.conv2(point1)
        point1 = self.conv3(point1)
        # # point1=point1+point2
        # point2=self.conv5(point1)
        point1 = point1 + x
        point1 = self.conv4(point1)
        # point1=self.sigmoid(point1)
        return x, point1


# class BAT(nn.Module):
#     def __init__(
#             self,
#             num_classes,
#             point_pred,
#             in_channels=512,
#             decoder=False,
#             transformer_type_index=0,
#             hidden_features=256,  # 256
#             number_of_query_positions=1,
#             segmentation_attention_heads=8):

#         super(BAT, self).__init__()

#         self.num_classes = num_classes
#         self.point_pred = point_pred
#         self.transformer_type = "BoundaryAwareTransformer" if transformer_type_index == 0 else "Transformer"
#         self.use_decoder = decoder

#         self.in_channels = in_channels

#         # self.convolution_mapping = nn.Conv2d(in_channels=in_channels,
#         #                                      out_channels=hidden_features,
#         #                                      kernel_size=(1, 1),
#         #                                      stride=(1, 1),
#         #                                      padding=(0, 0),
#         #                                      bias=True)

#         self.query_positions = nn.Parameter(data=torch.randn(
#             number_of_query_positions, hidden_features, dtype=torch.float),
#                                             requires_grad=True)

#         self.row_embedding = nn.Parameter(data=torch.randn(100,
#                                                            hidden_features //
#                                                            2,
#                                                            dtype=torch.float),
#                                           requires_grad=True)
#         self.column_embedding = nn.Parameter(data=torch.randn(
#             100, hidden_features // 2, dtype=torch.float),
#                                              requires_grad=True)

#         # self.transformer =BoundaryAwareTransformer(d_model=hidden_features,normalize_before=False,num_encoder_layers=6,num_decoder_layers=2,Atrous=False)
#         self.transformer =BoundaryAwareTransformer(d_model=hidden_features,normalize_before=False,num_decoder_layers=0,point_pred_layers=1,Atrous=False)

#         if self.use_decoder:
#             self.BCA = BoundaryCrossAttention(hidden_features, 8)

#         # self.trans_out_conv = nn.Conv2d(in_channels=hidden_features,
#         #                                 out_channels=in_channels,
#         #                                 kernel_size=(1, 1),
#         #                                 stride=(1, 1),
#         #                                 padding=(0, 0),
#         #                                 bias=True)

#     def forward(self, x):
#         h = x.size()[2]
#         w = x.size()[3]
#         feature_map = x
#         features=x
#         # features = self.convolution_mapping(feature_map)
#         height, width = features.shape[2:]
#         batch_size = features.shape[0]
#         positional_embeddings = torch.cat([
#             self.column_embedding[:height].unsqueeze(dim=0).repeat(
#                 height, 1, 1),
#             self.row_embedding[:width].unsqueeze(dim=1).repeat(1, width, 1)
#         ],dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1)

#         if self.transformer_type == 'BoundaryAwareTransformer':
#             latent_tensor, features_encoded, point_maps = self.transformer(
#                 features, None, self.query_positions, positional_embeddings)
#         else:
#             latent_tensor, features_encoded = self.transformer(
#                 features, None, self.query_positions, positional_embeddings)
#             point_maps = []

#         latent_tensor = latent_tensor.permute(2, 0, 1)
#         # shape:(bs, 1 , 128)

#         # if self.use_decoder:
#         #     features_encoded, point_dec = self.BCA(features_encoded,
#         #                                            latent_tensor)
#         #     point_maps.append(point_dec)

#         # trans_feature_maps = self.trans_out_conv(
#         #     features_encoded.contiguous())  #.contiguous()
#         trans_feature_maps = features_encoded.contiguous()
#         trans_feature_maps = trans_feature_maps + feature_map

#         output = F.interpolate(
#             trans_feature_maps, size=(h, w),
#             mode="bilinear")  # (shape: (batch_size, num_classes, h, w))

#         if self.point_pred == 1:
#             return output, point_maps[0]

#         return output

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    model = deeplabv3plus_pvtv2().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1 = model(input_tensor)
    print(prediction1.size())