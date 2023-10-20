from typing import Tuple

import torch
import torch.nn as nn

from tcn_lib.blocks.PointwiseLayer import PointwiseLayer
from tcn_lib.blocks.TemporalLayer import TemporalLayer


class TemporalBottleneck(nn.Module):

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
        """Basically the same idea as a bottleneck layer in a ResNet, but now
        for a 1D convolutional network.

        Usually n_channels is a tuple of (channels / expansion=4, channels).
        By setting groups > 1, we effectively use a ResNeXt bottleneck layer. Setting groups = -1,
        creates a depthwise separable convolution layer.
        """

        super(TemporalBottleneck, self).__init__()

        n_intermediates, n_outputs = n_channels
        requires_downsample = n_inputs != n_outputs and residual

        self.residual = residual

        self.temp_layer1 = PointwiseLayer(n_inputs, n_intermediates, dropout,
                                          batch_norm, weight_norm, True)
        self.temp_layer2 = TemporalLayer(n_intermediates, n_intermediates,
                                         kernel_size, stride, dilation,
                                         padding, dropout, batch_norm,
                                         weight_norm, True, n_intermediates if groups == -1 else groups)
        self.temp_layer3 = PointwiseLayer(n_intermediates, n_outputs, dropout,
                                          batch_norm, weight_norm, False)

        self.downsample = PointwiseLayer(
            n_inputs, n_outputs, dropout, batch_norm, weight_norm,
            False) if requires_downsample else None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        res = x

        out = self.temp_layer1(x)
        out = self.temp_layer2(out)
        out = self.temp_layer3(out)

        if self.residual:
            if self.downsample is not None:
                res = self.downsample(res)

            out += res

        out = self.relu(out)

        return out
