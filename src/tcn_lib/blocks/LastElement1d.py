from torch import nn


class LastElement1d(nn.Module):

    def __init__(self):
        super(LastElement1d, self).__init__()

    def forward(self, x):
        return x[:, :, -1]
