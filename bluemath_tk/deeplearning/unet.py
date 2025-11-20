"""
Unet module.

This module is will try to generalize models like the ones in:
https://github.com/oaeen/wind2iwp
"""

import torch.nn as nn

from .layers import DoubleConv, DoubleConv3D, Down, Down3D, OutConv, OutConv3D, Up, Up3D


class UNet(nn.Module):
    """
    U-Net architecture for 2D image segmentation/regression.

    Parameters
    ----------
    n_channels : int
        Number of input channels.
    n_classes : int
        Number of output channels/classes.
    base_channels : int, optional
        Base number of channels. Default is 64.
    bilinear : bool, optional
        Whether to use bilinear upsampling. Default is True.
    """

    def __init__(self, n_channels=1, n_classes=1, base_channels=64, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet3D(nn.Module):
    """
    U-Net architecture for 3D volumetric data.

    Parameters
    ----------
    n_channels : int
        Number of input channels.
    n_classes : int
        Number of output channels/classes.
    base_channels : int, optional
        Base number of channels. Default is 16.
    trilinear : bool, optional
        Whether to use trilinear upsampling. Default is True.
    """

    def __init__(self, n_channels=1, n_classes=1, base_channels=16, trilinear=True):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = DoubleConv3D(n_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)
        factor = 2 if trilinear else 1
        self.down4 = Down3D(base_channels * 8, base_channels * 16 // factor)
        self.up1 = Up3D(base_channels * 16, base_channels * 8 // factor, trilinear)
        self.up2 = Up3D(base_channels * 8, base_channels * 4 // factor, trilinear)
        self.up3 = Up3D(base_channels * 4, base_channels * 2 // factor, trilinear)
        self.up4 = Up3D(base_channels * 2, base_channels, trilinear)
        self.outc = OutConv3D(base_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
