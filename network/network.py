import torch.nn as nn
from .ResBlock import ResBlock


"""
    Returns a residual block of 2 convolutional layers as used in the paper (Figure 2). Downsampling will automatically apply if inc != outc.
"""
def PaperBlock(inc, outc, residual):
    # Inner layers according to the paper. The last ReLU happens after the residual is applied (in the ResBlock class)
    inner_layers = [
        nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, padding=1, stride=(1 if inc==outc else 2)),
        nn.BatchNorm2d(outc),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1),
        nn.BatchNorm2d(outc),
    ]

    if residual:
        # Turn layers into a network and turn that into a residual block
        inner = nn.Sequential(*inner_layers)
        return ResBlock(inner, relu=True)
    else:
        # Add the missing ReLU layer and return without the residual block
        inner_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*inner_layers)

class ResNet(nn.Module):

    def __init__(self, n, residual):
        super().__init__()

        # initial conv layer
        layers = [
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        ]

        # stack of 6n conv layers
        for filters in (16, 32, 64):
            for i in range(n):
                layers.append(
                    PaperBlock(
                        inc=(filters if i>0 or filters==16 else int(filters/2)),     # on the first block of a given filter number we're coming from a higher filter number (except for the very first one)
                        outc=filters,
                        residual=residual,
                    )
                )

        # final classification
        layers.append(nn.AvgPool2d(kernel_size=8))      # our final feature maps are of size 8, so an equal kernel_size of 8 will yield global avg pooling
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=64, out_features=10))

        # Put network together
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


    
