"""File contents taken from:

https://github.com/locuslab/TCN/blob/2f8c2b817050206397458d
fd1f5a25ce8a32fe65/TCN/tcn.py.
"""

import torch
import torch.nn as nn


class Chomp1d(nn.Module):

    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor):
        if self.chomp_size == 0:
            return x

        # Cut off elements from the end of the sequence that extend beyond the input sequence length
        return x[:, :, :-self.chomp_size].contiguous()
