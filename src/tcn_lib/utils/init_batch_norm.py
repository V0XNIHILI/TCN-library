from typing import Union

import torch.nn as nn

def init_batch_norm(module: nn.BatchNorm1d):
    nn.init.constant_(module.weight, 1)
    nn.init.constant_(module.bias, 0)
