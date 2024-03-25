"""File contents taken from:

https://github.com/locuslab/TCN/blob/2f8c2b817050206397458d
fd1f5a25ce8a32fe65/TCN/tcn.py.
"""

from typing import List, Tuple, Union

import torch.nn as nn

from tcn_lib.blocks import TemporalBlock, TemporalBottleneck
from tcn_lib.utils import init_tcn_conv_weight, init_batch_norm

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
                 residual=True,
                 zero_init_residual=False):
        if zero_init_residual == True and not batch_norm:
            raise ValueError(
                "To use zero_init_residual, batch_norm has to be set to True."
            )
    
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

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init_tcn_conv_weight(m)
            elif isinstance(m, nn.BatchNorm1d):
                init_batch_norm(m)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, TemporalBlock):
                    nn.init.constant_(m.temp_layer2[1].weight, 0)
                elif isinstance(m, TemporalBottleneck):
                    nn.init.constant_(m.temp_layer3[1].weight, 0)
