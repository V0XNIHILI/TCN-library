from typing import List, Tuple, Union, Final, Optional
import warnings

import torch
from torch import nn

from tcn_lib.blocks import LastElement1d, TemporalConvNet
from tcn_lib.stats import get_receptive_field_size, get_kernel_size_and_layers


class TCN(nn.Module):

    has_linear_layer: Final[bool]
    transpose_input: Final[bool]

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 channel_sizes: Union[List[Union[int, Tuple[int, int]]], int, Tuple[int, int]],
                 kernel_size: Optional[Union[int, List[int]]] = None,
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
                 input_length: Optional[int] = None,
                 crop_hidden_states: bool = False,
                 transpose_input: bool = False):
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
            crop_hidden_states (bool, optional): Whether to crop the hidden states to the minimum required length; requires input_length to be specified. Defaults to False.
            transpose_input (bool, optional): Whether to transpose the input from (N, L_in, C_in) to (N, C_in, L_in). Defaults to False.
        """

        super(TCN, self).__init__()

        if kernel_size is None:
            assert input_length is not None, "If kernel_size is not specified, input_length must be specified."
            assert type(channel_sizes) is int or type(channel_sizes) is tuple, "If kernel_size is not specified but input_length is, channel_sizes must be an int or tuple of ints."

            k, l = get_kernel_size_and_layers(input_length)

            kernel_size = [k] * l

        if type(channel_sizes) is int:
            channel_sizes = [(channel_sizes, channel_sizes)] * len(kernel_size)
        elif type(channel_sizes) is tuple:
            channel_sizes = [channel_sizes] * len(kernel_size)
        else:
            # Make sure that also specifying one channel size per temporal layer works
            channel_sizes = [channel_size if type(channel_size) is not int else (channel_size, channel_size) for channel_size in channel_sizes]

        self.pad_inputs = None

        if crop_hidden_states:
            assert input_length is not None, "If crop_hidden_states is set to True, input_length must be specified."
            assert take_last_element is True, "Cropping the hidden states only works when the last element is taken. Set take_last_element to True."

        if input_length is not None:
            receptive_field_size = get_receptive_field_size(kernel_size, len(channel_sizes))

            if input_length > receptive_field_size:
                warnings.warn(f"Input length ({input_length}) is larger than the receptive field size ({receptive_field_size}). Use get_kernel_size_and_layers({input_length}) to find the kernel size and number of layers that have a receptive field size closest to the input length.")

            if crop_hidden_states:
                padding = receptive_field_size - input_length

                self.pad_inputs = nn.ZeroPad1d((padding, 0))

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
                              zero_init_residual=zero_init_residual,
                              crop_hidden_states=crop_hidden_states)

        if take_last_element:
            self.embedder = nn.Sequential(tcn, LastElement1d())
        else:
            self.embedder = tcn

        self.has_linear_layer = output_size != -1

        if self.has_linear_layer:
            self.fc = nn.Linear(channel_sizes[-1][1], output_size)

        self.transpose_input = transpose_input

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TCN.

        Args:
            inputs (torch.Tensor): Inputs into the TCN. Tensor should be of shape (N = batch size, C_in = input channels, L_in = input length) or (N, L_in, C_in) if transpose_input is True.

        Returns:
            torch.Tensor: Output of the TCN. Tensor will be of shape (N, C_out).
        """

        if self.transpose_input:
            inputs = inputs.transpose(1, 2)

        if self.pad_inputs:
            inputs = self.pad_inputs(inputs)

        out = self.embedder(inputs)

        if self.has_linear_layer:
            og_shape = None
            
            if len(out.shape) == 3:
                # out shape = (N, C_out, L_out)
                og_shape = out.shape
                # out shape = (N * L_out, C_out), since nn.Linear expects the last dimension to be the channel dimension, currently that is the sequence length dimension
                out = out.view(-1, out.shape[1])
    
            out = self.fc(out)

            if og_shape is not None:
                # Reshape back to (N, C_out, L_out) for CE loss, which expects (N, C, ...)
                out = out.view(*og_shape)

        return out
