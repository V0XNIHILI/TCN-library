"""File contents taken from:

https://github.com/locuslab/TCN/blob/2f8c2b817050206397458d
fd1f5a25ce8a32fe65/TCN/tcn.py.
"""

from typing import Tuple

import torch
import torch.nn as nn

from tcn_lib.blocks.PointwiseLayer import PointwiseLayer
from tcn_lib.blocks.TemporalLayer import TemporalLayer


class TemporalBlock(nn.Module):

    def __init__(self,
                 n_inputs: int,
                 n_channels: Tuple[int, int],
                 kernel_size: int,
                 stride: int,
                 dilation: int,
                 padding: int,
                 dropout=0.2,
                 batch_norm=False,
                 weight_norm=False,
                 groups=1,
                 residual=True):
        super(TemporalBlock, self).__init__()

        n_intermediates, n_outputs = n_channels
        requires_downsample = n_inputs != n_outputs and residual

        self.residual = residual

        self.temp_layer1 = TemporalLayer(n_inputs, n_intermediates,
                                         kernel_size, stride, dilation,
                                         padding, dropout, batch_norm,
                                         weight_norm, True, n_inputs if groups == -1 else groups)
        self.temp_layer2 = TemporalLayer(n_intermediates, n_outputs,
                                         kernel_size, 1, dilation, padding,
                                         dropout, batch_norm, weight_norm,
                                         False, n_intermediates if groups == -1 else groups)

        # No bias needed in this layer as the bias of temp_layer2 will have the same effect
        self.downsample = PointwiseLayer(
            n_inputs, n_outputs, dropout, batch_norm, weight_norm,
            False, True) if requires_downsample else None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        out = self.temp_layer1(x)
        out = self.temp_layer2(out)

        if self.residual:
            res = x

            if self.downsample is not None:
                res = self.downsample(res)   
                     
            out += res

        out = self.relu(out)

        return out
