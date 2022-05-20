import torch.nn.functional as F
import torch.nn as nn
import torch


class _simple_learner(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Conv2d(d_model * 2, d_model, 1, 1)

    def forward(self, f_low, f_high):
        low_size = f_low.shape[2:]
        f2_high = F.interpolate(f_high, size=low_size)

        f2_low = torch.cat([f_low, f2_high], dim=1)
        f2_low = self.mlp(f2_low)
        return f2_low


class xboundlearnerv2(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0):
        super().__init__()

        self.xbl = xboundlearner(d_model,
                                 nhead,
                                 dim_feedforward=dim_feedforward,
                                 dropout=dropout)
        self.xbl1 = xboundlearner(d_model,
                                  nhead,
                                  dim_feedforward=dim_feedforward,
                                  dropout=dropout)
        self.mlp = nn.Conv2d(d_model * 2, d_model, 1, 1)

    def forward(self, f_low, f_high, xi_low, xi_high):
        f2_low = self.xbl(f_low, xi_high)
        f2_high = self.xbl1(f_high, xi_low)

        low_size = f2_low.shape[2:]
        f2_high = F.interpolate(f2_high, size=low_size)

        f2_low = torch.cat([f2_low, f2_high], dim=1)
        f2_low = self.mlp(f2_low)
        return f2_low + f2_low


class xboundlearner(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(d_model,
                                                nhead,
                                                dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU()

    def forward(self, tgt, src):
        "tgt shape: Batch_size, C, H, W "
        "src shape: Batch_size, 1, C    "

        B, C, h, w = tgt.shape
        tgt = tgt.view(B, C, h * w).permute(2, 0, 1)  # shape: L, B, C

        src = src.permute(1, 0, 2)  # shape: Q:1, B, C

        fusion_feature = self.cross_attn(query=tgt, key=src, value=src)[0]
        tgt = tgt + self.dropout1(fusion_feature)
        tgt = self.norm1(tgt)

        tgt1 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt1)
        tgt = self.norm2(tgt)
        return tgt.permute(1, 2, 0).view(B, C, h, w)


class BoundaryWiseAttentionGateAtrous2D(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):

        super(BoundaryWiseAttentionGateAtrous2D, self).__init__()

        modules = []

        if hidden_channels == None:
            hidden_channels = in_channels // 2

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True)))
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels,
                          hidden_channels,
                          3,
                          padding=1,
                          dilation=1,
                          bias=False), nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)))
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels,
                          hidden_channels,
                          3,
                          padding=2,
                          dilation=2,
                          bias=False), nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)))
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels,
                          hidden_channels,
                          3,
                          padding=4,
                          dilation=4,
                          bias=False), nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)))
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels,
                          hidden_channels,
                          3,
                          padding=6,
                          dilation=6,
                          bias=False), nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)))

        self.convs = nn.ModuleList(modules)

        self.conv_out = nn.Conv2d(5 * hidden_channels, 1, 1, bias=False)

    def forward(self, x):
        " x.shape: B, C, H, W "
        " return: feature, weight (B,C,H,W) "
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        weight = torch.sigmoid(self.conv_out(res))
        x = x * weight + x
        return x, weight


class BoundaryWiseAttentionGate2D(nn.Sequential):
    def __init__(self, in_channels, hidden_channels=None):
        super(BoundaryWiseAttentionGate2D, self).__init__(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False), nn.Conv2d(in_channels, 1, kernel_size=1))

    def forward(self, x):
        " x.shape: B, C, H, W "
        " return: feature, weight (B,C,H,W) "
        weight = torch.sigmoid(
            super(BoundaryWiseAttentionGate2D, self).forward(x))
        x = x * weight + x
        return x, weight
