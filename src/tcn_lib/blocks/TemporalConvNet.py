"""File contents taken from:

https://github.com/locuslab/TCN/blob/2f8c2b817050206397458d
fd1f5a25ce8a32fe65/TCN/tcn.py.
"""

from typing import List, Tuple

import torch.nn as nn

from tcn_lib.blocks import TemporalBlock, TemporalBottleneck


class TemporalConvNet(nn.Sequential):

    def __init__(self,
                 num_inputs: int,
                 num_channels: List[Tuple[int, int]],
                 kernel_size=2,
                 dropout=0.2,
                 batch_norm=False,
                 weight_norm=False,
                 residual=True):
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i

            in_channels = num_inputs if i == 0 else num_channels[i - 1][1]
            inside_channels = num_channels[i]

            layers += [
                TemporalBlock(in_channels,
                              inside_channels,
                              kernel_size,
                              stride=1,
                              dilation=dilation_size,
                              padding=(kernel_size - 1) * dilation_size,
                              dropout=dropout,
                              batch_norm=batch_norm,
                              weight_norm=weight_norm,
                              residual=residual)
            ]

        super(TemporalConvNet, self).__init__(*layers)
