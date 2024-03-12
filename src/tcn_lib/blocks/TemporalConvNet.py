"""File contents taken from:

https://github.com/locuslab/TCN/blob/2f8c2b817050206397458d
fd1f5a25ce8a32fe65/TCN/tcn.py.
"""

from typing import List, Tuple, Union

import torch.nn as nn

from tcn_lib.blocks import TemporalBlock, TemporalBottleneck


class TemporalConvNet(nn.Sequential):

    def __init__(self,
                 num_inputs: int,
                 num_channels: List[Tuple[int, int]],
                 kernel_size: Union[int, List[int]],
                 dropout=0.2,
                 batch_norm=False,
                 weight_norm=False,
                 bottleneck=False,
                 groups=1,
                 residual=True):
        layers = []

        Block = TemporalBottleneck if bottleneck else TemporalBlock

        for i in range(len(num_channels)):
            dilation_size = 2**i

            in_channels = num_inputs if i == 0 else num_channels[i - 1][1]
            block_kernel_size = kernel_size[i] if isinstance(kernel_size, list) else kernel_size

            layers += [
                Block(in_channels,
                      num_channels[i],
                      block_kernel_size,
                      stride=1,
                      dilation=dilation_size,
                      padding=(block_kernel_size - 1) * dilation_size,
                      dropout=dropout,
                      batch_norm=batch_norm,
                      weight_norm=weight_norm,
                      groups=groups,
                      residual=residual)
            ]

        super(TemporalConvNet, self).__init__(*layers)
