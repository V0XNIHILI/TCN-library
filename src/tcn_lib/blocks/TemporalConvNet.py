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
                 dropout_mode: str = 'standard',
                 batch_norm: bool = False,
                 weight_norm: bool = False,
                 bottleneck: bool = False,
                 groups: int = 1,
                 residual: bool = True,
                 force_downsample: bool = False,
                 zero_init_residual: bool = False,
                 crop_hidden_states: bool =False):
        if zero_init_residual == True and not batch_norm:
            raise ValueError(
                "To use zero_init_residual, batch_norm has to be set to True."
            )
    
        layers = []

        Block = TemporalBottleneck if bottleneck else TemporalBlock

        if crop_hidden_states == True:
            assert not bottleneck, "Bottleneck layers are not supported when crop_hidden_states is set to True."

        for i in range(len(num_channels)):
            in_channels = num_inputs if i == 0 else num_channels[i - 1][1]
            block_kernel_size = kernel_size[i] if isinstance(kernel_size, list) else kernel_size
            
            if crop_hidden_states:
                dilation_size = 1
                padding = 0
                stride = (1, 2)
            else:
                dilation_size = 2**i
                padding = (block_kernel_size - 1) * dilation_size
                stride = (1, 1)

            layers += [
                Block(in_channels,
                      num_channels[i],
                      block_kernel_size,
                      stride=stride,
                      dilation=dilation_size,
                      padding=padding,
                      dropout=dropout,
                      dropout_mode=dropout_mode,
                      batch_norm=batch_norm,
                      weight_norm=weight_norm,
                      groups=groups,
                      residual=residual,
                      force_downsample=force_downsample)
            ]

        super(TemporalConvNet, self).__init__(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Initialize weights following Kaiming He's scheme
                init_tcn_conv_weight(m)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize BatchNorm following PyTorch's ResNet
                # implementation: https://github.com/pytorch/vision/blob/0d68c7df8640abff43355afd57c494cf5d74f4a9/torchvision/models/resnet.py#L211
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
