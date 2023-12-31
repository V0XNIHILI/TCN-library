from typing import List, Tuple, Union, Final

import torch
from torch import nn

from tcn_lib.blocks import LastElement1d, TemporalConvNet


class TCN(nn.Module):

    has_linear_layer : Final[bool]

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 channel_sizes: Union[List[int], List[Tuple[int, int]]],
                 kernel_size: int,
                 dropout: float = 0.0,
                 batch_norm: bool = False,
                 weight_norm: bool = False,
                 bottleneck: bool = False,
                 groups: int = 1,
                 residual: bool = True):
        """Temporal Convolutional Network. Implementation based off of: 
        https://github.com/locuslab/TCN/blob/master/TCN/mnist_pixel/model.py.

        Args:
            input_size (int): Dimensionality of each input time step.
            output_size (int): Final output size. Set to -1 to omit the linear layer.
            channel_sizes (Union[List[int], List[Tuple[int, int]]]): Number of channels in each layer.
            kernel_size (int): Kernel size (the same for each layer)
            dropout (float, optional): Dropout probability for the temporal convolutional layers. Defaults to 0.0.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            weight_norm (bool, optional): Whether to use weight normalization. Defaults to False.
            bottleneck (bool, optional): Whether to use bottleneck layers. Note that when bottleneck = True and groups = -1, a depthwise separable convolution is created. Defaults to False.
            groups (int, optional): Number of groups for the temporal convolutional layers. Set to -1 for depthwise convolutions. Defaults to 1.
            residual (bool, optional): Whether to use residual connections. Defaults to True.
        """

        super(TCN, self).__init__()

        # Make sure that also specifying one channel size per temporal layer works
        if type(channel_sizes[0]) == int:
            channel_sizes = [[a, a] for a in channel_sizes]

        self.embedder = nn.Sequential(
            TemporalConvNet(input_size,
                            channel_sizes,
                            kernel_size=kernel_size,
                            dropout=dropout,
                            batch_norm=batch_norm,
                            weight_norm=weight_norm,
                            bottleneck=bottleneck,
                            groups=groups,
                            residual=residual), LastElement1d())

        self.has_linear_layer = output_size != -1

        if self.has_linear_layer:
            self.linear = nn.Linear(channel_sizes[-1][1], output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Inputs need to have shape (N, C_in, L_in)"""

        out = self.embedder(inputs)

        if self.has_linear_layer:
            out = self.linear(out)

        return out
