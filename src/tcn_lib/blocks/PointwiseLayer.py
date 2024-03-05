import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from tcn_lib.utils import conditional_apply, init_tcn_conv_weight


class PointwiseLayer(nn.Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout=0.2,
                 batch_norm: bool = False,
                 use_weight_norm=False,
                 with_activation=True,
                 without_bias=True):
        conv = conditional_apply(weight_norm, use_weight_norm)(
            nn.Conv1d(
                in_channels,
                out_channels,
                1,
                # Following: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm
                bias=not (batch_norm or without_bias)))
        normalize = nn.BatchNorm1d(
            out_channels) if batch_norm else nn.Identity()
        relu = nn.ReLU(inplace=True) if with_activation else nn.Identity()
        dropout = nn.Dropout(dropout)

        init_tcn_conv_weight(conv)

        super(PointwiseLayer, self).__init__(conv, normalize, relu, dropout)
