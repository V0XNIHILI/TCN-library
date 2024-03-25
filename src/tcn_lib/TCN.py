from typing import List, Tuple, Union, Final

import torch
from torch import nn

from tcn_lib.blocks import LastElement1d, TemporalConvNet


class TCN(nn.Module):

    has_linear_layer : Final[bool]

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 channel_sizes: List[Union[int, Tuple[int, int]]],
                 kernel_size: Union[int, List[int]],
                 dropout: float = 0.0,
                 batch_norm: bool = False,
                 weight_norm: bool = False,
                 bottleneck: bool = False,
                 groups: int = 1,
                 residual: bool = True,
                 zero_init_residual: bool = False):
        """Temporal Convolutional Network. Implementation based off of: 
        https://github.com/locuslab/TCN/blob/master/TCN/mnist_pixel/model.py.

        Args:
            input_size (int): Dimensionality or number of channels of each input time step.
            output_size (int): Final output size. Set to -1 to omit the linear layer.
            channel_sizes (Union[List[int], List[Tuple[int, int]]]): Number of channels in each layer.
            kernel_size (Union[int, List[int]]): Kernel size. Can be specified for the whole network as a single int, per layer as a list of ints.
            dropout (float, optional): Dropout probability for the temporal convolutional layers. Defaults to 0.0.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            weight_norm (bool, optional): Whether to use weight normalization. Defaults to False.
            bottleneck (bool, optional): Whether to use bottleneck layers. Note that when bottleneck = True and groups = -1, a depthwise separable convolution is created. Defaults to False.
            groups (int, optional): Number of groups for the temporal convolutional layers. Set to -1 for depthwise convolutions. Defaults to 1.
            residual (bool, optional): Whether to use residual connections. Defaults to True.
            zero_init_residual (bool, optional): Whether to zero initialize the residual connections (per: https://arxiv.org/abs/1706.0267). Defaults to False.
        """

        super(TCN, self).__init__()

        # Make sure that also specifying one channel size per temporal layer works
        channel_sizes = [channel_size if type(channel_size) is not int else (channel_size, channel_size) for channel_size in channel_sizes]

        self.embedder = nn.Sequential(
            TemporalConvNet(input_size,
                            channel_sizes,
                            kernel_size=kernel_size,
                            dropout=dropout,
                            batch_norm=batch_norm,
                            weight_norm=weight_norm,
                            bottleneck=bottleneck,
                            groups=groups,
                            residual=residual,
                            zero_init_residual=zero_init_residual), LastElement1d())

        self.has_linear_layer = output_size != -1

        if self.has_linear_layer:
            self.fc = nn.Linear(channel_sizes[-1][1], output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TCN.

        Args:
            inputs (torch.Tensor): Inputs into the TCN. Tensor should be of shape (N = batch size, C_in = input channels, L_in = input length).

        Returns:
            torch.Tensor: Output of the TCN. Tensor will be of shape (N, C_out).
        """

        out = self.embedder(inputs)

        if self.has_linear_layer:
            out = self.fc(out)

        return out
