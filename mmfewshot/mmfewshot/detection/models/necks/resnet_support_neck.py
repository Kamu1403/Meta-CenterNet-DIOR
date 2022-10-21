# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmdet.models.roi_heads import ResLayer
from torch import Tensor
from mmdet.models.backbones import ResNet

from mmdet.models.builder import NECKS


@NECKS.register_module()
class MetaCenterNetSupportNeck(ResLayer):
    """Shared resLayer for metarcnn and fsdetview.

    It provides different forward logics for query and support images.
    """

    def __init__(self, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        block, stage_blocks = ResNet.arch_settings[kwargs['depth']]
        inplanes = 64 * 2 ** (kwargs['stage']) * block.expansion
        self.avg_group_size = inplanes // out_channels
        self.avg_inplanes = inplanes
        self.with_avg = True
        if self.avg_group_size == 1:
            self.with_avg = False

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for query images.

        Args:
            x (Tensor): Features from backbone with shape (N, C, H, W).

        Returns:
            Tensor: Shape of (N, C).
        """
        res_layer = getattr(self, f'layer{self.stage + 1}')
        out = res_layer(x)
        if self.with_avg:
            out = torch.cat(([torch.mean(out[:, i:i + self.avg_group_size], dim=1, keepdim=True) for i in
                              range(0, self.avg_inplanes, self.avg_group_size)]), dim=1)
        return out
