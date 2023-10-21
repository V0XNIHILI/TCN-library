import torch.nn as nn


def init_tcn_conv_weight(conv: nn.Conv1d):
    nn.init.kaiming_uniform_(conv.weight,
                             mode='fan_in',
                             nonlinearity='relu')
