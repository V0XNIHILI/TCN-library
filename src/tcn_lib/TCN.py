from typing import List, Tuple, Union, Final, Optional
import warnings

import torch
from torch import nn

from tcn_lib.blocks import LastElement1d, TemporalConvNet
from tcn_lib.stats import get_receptive_field_size


class TCN(nn.Module):

    has_linear_layer: Final[bool]

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 channel_sizes: List[Union[int, Tuple[int, int]]],
                 kernel_size: Union[int, List[int]],
                 dropout: float = 0.0,
                 dropout_mode: str = 'standard',
                 batch_norm: bool = False,
                 weight_norm: bool = False,
                 bottleneck: bool = False,
                 groups: int = 1,
                 residual: bool = True,
                 force_downsample: bool = False,
                 zero_init_residual: bool = False,
                 take_last_element: bool = True,
                 input_length: Optional[int] = None):
        """Temporal Convolutional Network. Implementation based off of: 
        https://github.com/locuslab/TCN/blob/master/TCN/mnist_pixel/model.py.

        Args:
            input_size (int): Dimensionality or number of channels of each input time step.
            output_size (int): Final output size. Set to -1 to omit the linear layer.
            channel_sizes (Union[List[int], List[Tuple[int, int]]]): Number of channels in each layer.
            kernel_size (Union[int, List[int]]): Kernel size. Can be specified for the whole network as a single int, per layer as a list of ints.
            dropout (float, optional): Dropout probability for the temporal convolutional layers. Defaults to 0.0.
            dropout_mode (str, optional): Dropout mode. Can be 'standard' or '1d'. Defaults to 'standard'.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            weight_norm (bool, optional): Whether to use weight normalization. Defaults to False.
            bottleneck (bool, optional): Whether to use bottleneck layers. Note that when bottleneck = True and groups = -1, a depthwise separable convolution is created. Defaults to False.
            groups (int, optional): Number of groups for the temporal convolutional layers. Set to -1 for depthwise convolutions. Defaults to 1.
            residual (bool, optional): Whether to use residual connections. Defaults to True.
            force_downsample (bool, optional): Whether to force downsample in every layer instead of doing an identity shortcut when the number of input channels is equal to the number of output channels. Defaults to False.
            zero_init_residual (bool, optional): Whether to zero initialize the residual connections (per: https://arxiv.org/abs/1706.0267). Defaults to False.
            take_last_element (bool, optional): Whether to take the last element of the output. Defaults to True.
            input_length (Optional[int], optional): Length of the input; only used to check compatibility with the receptive field size. Defaults to None.
        """

        super(TCN, self).__init__()

        # Make sure that also specifying one channel size per temporal layer works
        channel_sizes = [channel_size if type(channel_size) is not int else (channel_size, channel_size) for channel_size in channel_sizes]

        if input_length is not None:
            receptive_field_size = get_receptive_field_size(kernel_size, len(channel_sizes))

            if input_length > receptive_field_size:
                warnings.warn(f"Input length ({input_length}) is larger than the receptive field size ({receptive_field_size}). Use get_kernel_size_and_layers({input_length}) to find the kernel size and number of layers that have a receptive field size closest to the input length.")

        tcn = TemporalConvNet(input_size,
                              channel_sizes,
                              kernel_size=kernel_size,
                              dropout=dropout,
                              dropout_mode=dropout_mode,
                              batch_norm=batch_norm,
                              weight_norm=weight_norm,
                              bottleneck=bottleneck,
                              groups=groups,
                              residual=residual,
                              force_downsample=force_downsample,
                              zero_init_residual=zero_init_residual)

        if take_last_element:
            self.embedder = nn.Sequential(tcn, LastElement1d())
        else:
            self.embedder = tcn

        self.has_linear_layer = output_size != -1

        if self.has_linear_layer:
            self.fc = nn.Linear(channel_sizes[-1][1], output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TCN.

        Args:
            inputs (torch.Tensor): Inputs into the TCN. Tensor should be of shape (N = batch size, C_in = input channels, L_in = input length)
                                   or can be of shape (N = batch size, L_in = input length) if C_in = 1.

        Returns:
            torch.Tensor: Output of the TCN. Tensor will be of shape (N, C_out).
        """

        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)

        out = self.embedder(inputs)

        if self.has_linear_layer:
            out = self.fc(out)

        return out
