from typing import Optional

import torch.nn as nn

from tcn_lib.blocks.GRELU import GRELU


def select_act_fn(act: Optional[str]) -> nn.Module:
    if act is None:
        return nn.Identity()

    if act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'silu':
        return nn.SiLU(inplace=True)
    elif act == 'grelu':
        return GRELU()
    else:
        raise ValueError(f"Unsupported activation function: {act}. Supported values are 'relu', 'gelu', 'silu', 'grelu', and None but got {act}.")
