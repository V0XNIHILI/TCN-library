"""File contents taken from:

https://github.com/locuslab/TCN/blob/2f8c2b817050206397458d
fd1f5a25ce8a32fe65/TCN/tcn.py.
"""

import torch.nn as nn

from tcn_lib.blocks.Chomp1d import Chomp1d
from tcn_lib.utils import conditional_apply


class TemporalLayer(nn.Sequential):

    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int,
                 padding: int,
                 dropout=0.2,
                 batch_norm=False,
                 use_weight_norm=False,
                 with_activation=True,
                 groups=1):
        # We apply padding on both sides and remove the extra values on the right side
        # We could have also only applied padding on the left side like here:
        # https://github.com/locuslab/TCN/issues/8#issuecomment-384345206
        conv = conditional_apply(nn.utils.parametrizations.weight_norm, use_weight_norm)(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                # Following:
                # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm
                bias=not batch_norm))
        chomp = Chomp1d(padding)
        normalize = nn.BatchNorm1d(n_outputs) if batch_norm else nn.Identity()
        relu = nn.ReLU(inplace=True) if with_activation else nn.Identity()
        dropout = nn.Dropout(dropout)

        super(TemporalLayer, self).__init__(conv, normalize, chomp, relu,
                                            dropout)

