# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from torch.utils.checkpoint import checkpoint
# from mmdet3d.models.backbones.resnet import ConvModule
# from mmdet.models import NECKS


class FPN_LSS(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        ) if use_input_conv else None
        if use_input_conv:
            in_channels = out_channels * channels_factor
        self.conv34 = nn.Sequential(
            nn.Conv2d(
                in_channels[1],
                out_channels[1] * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels[1] * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels[1] * channels_factor,
                out_channels[1] * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels[1] * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        self.conv23 = nn.Sequential(
            nn.Conv2d(
                in_channels[0],
                out_channels[0] * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels[0] * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels[0] * channels_factor,
                out_channels[0] * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels[0] * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=extra_upsample,
                    mode='bilinear',
                    align_corners=True),
                nn.Conv2d(
                    out_channels * channels_factor,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(
                    lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        x2, x3, x4 = feats # [B, 64, 160, 160] [B, 128, 80, 80] [B, 256, 40, 40]
        x2 = x2.dense()
        x3 = x3.dense()
        x4 = x4.dense()

        x4 = self.up(x4) # [B, 256, 80, 80]

        x34 = torch.cat([x3, x4], dim=1) # [B, 384, 80, 80]
        x34 = self.conv34(x34) # [B, 128, 80, 80]
        x34 = self.up(x34) # [B, 128, 160, 160]

        if x34.shape[-2:] != x2.shape[-2:]:
            x34 = nn.functional.interpolate(x34, size=x2.shape[-2:])

        x23 = torch.cat([x2, x34], dim=1) # [B, 192, 160, 160]
        x23 = self.conv23(x23) # [B, 128, 160, 160]

        return x23
