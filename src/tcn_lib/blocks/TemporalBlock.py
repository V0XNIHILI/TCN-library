"""File contents taken from:

https://github.com/locuslab/TCN/blob/2f8c2b817050206397458d
fd1f5a25ce8a32fe65/TCN/tcn.py.
"""

from typing import Tuple, Union

import torch
import torch.nn as nn

from tcn_lib.blocks.PointwiseLayer import PointwiseLayer
from tcn_lib.blocks.TemporalLayer import TemporalLayer


class TemporalBlock(nn.Module):

    def __init__(self,
                 n_inputs: int,
                 n_channels: Tuple[int, int],
                 kernel_size: int,
                 stride: Union[int, Tuple[int, int]],
                 dilation: int,
                 padding: int,
                 dropout: float = 0.2,
                 dropout_mode: str = 'standard',
                 batch_norm: bool = False,
                 weight_norm: bool = False,
                 groups: int = 1,
                 residual: bool = True,
                 force_downsample: bool = False):
        super(TemporalBlock, self).__init__()

        n_intermediates, n_outputs = n_channels
        requires_downsample = (n_inputs != n_outputs or force_downsample) and residual

        self.residual = residual

        if isinstance(stride, int):
            stride = (stride, stride)

        self.temp_layer1 = TemporalLayer(n_inputs, n_intermediates,
                                         kernel_size, stride[0], dilation,
                                         padding, dropout, dropout_mode, batch_norm,
                                         weight_norm, True, n_inputs if groups == -1 else groups)
        self.temp_layer2 = TemporalLayer(n_intermediates, n_outputs,
                                         kernel_size, stride[1], dilation, padding,
                                         dropout, dropout_mode, batch_norm, weight_norm,
                                         False, n_intermediates if groups == -1 else groups)

        if requires_downsample:
            # No bias needed in this layer as the bias of temp_layer2 will have the same effect
            self.downsample = PointwiseLayer(
                n_inputs, n_outputs, stride[1],
                dropout, batch_norm, weight_norm,
                False, False)
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        out = self.temp_layer1(x)
        out = self.temp_layer2(out)

        if self.residual:
            res = x

            if self.downsample is not None:
                k = self.temp_layer1[0].kernel_size[0]
                stride = self.temp_layer2[0].stride[0]

                if stride != 1:
                    res = res[:, :, 2*k-2:]

                res = self.downsample(res)
            else:
                stride = self.temp_layer2[0].stride[0]
                k = self.temp_layer1[0].kernel_size[0]
                if stride != 1:
                    res = res[:, :, 2*k-2::2]
        
            out += res

        out = self.relu(out)

        return out
