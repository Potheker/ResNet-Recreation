import torch.nn as nn
from math import floor, ceil

"""
    A class representing a sequence of layers, with residual learning implemented across itself. Dimension change is handled with option (A) (zero padding) from the paper.

    Arguments:
        inner (nn.Module): The sequence of layers inside the block
"""
class ResBlock(nn.Module):
    def __init__(self, inner: nn.Module, relu):
        super().__init__()
        self.inner = inner
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        y = self.inner(x)

        identity = x
        # Downsampling
        factor = [1, 1]
        # Check if dimensions can be downsampled
        for i in range(2):
            s = identity.shape[2+i] / y.shape[2+i]
            if s.is_integer():
                factor[i] = int(s)
            else:
                raise Exception(f"Dimension {identity.shape} can't be downsampled to {y.shape}")
        # Do downsampling
        if not factor == [1, 1]:
            identity = nn.functional.avg_pool2d(identity, factor, factor)

        # Channel matching (zero-padding)
        add_channels = y.shape[1] - identity.shape[1]
        pad_left = int(add_channels/2) if add_channels%2 == 0 else floor(add_channels/2)
        pad_right = int(add_channels/2) if add_channels%2 == 0 else ceil(add_channels/2)

        if add_channels > 0:
            identity = nn.functional.pad(identity, (0,0,  0,0,  pad_left,pad_right))

        if self.relu:
            return self.relu(y+identity)
        else:
            return y+identity