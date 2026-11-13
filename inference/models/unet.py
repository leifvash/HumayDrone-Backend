"""
U-Net architecture for pixel-level canopy discoloration segmentation.
Directly implements Section 3.2.3.9 of the Capstone paper.
"""
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

if nn is not None:
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv(x)

    class RiceCanopyUNet(nn.Module):
        """
        U-Net network predicting binary discoloration masks:
        0: Healthy Canopy / Background
        1: Discolored Lesion (Bacterial Leaf Blight / White Head)
        """
        def __init__(self, in_channels=3, out_channels=1):
            super().__init__()
            self.inc = DoubleConv(in_channels, 32)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
            self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

            self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.conv1 = DoubleConv(256, 128)

            self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.conv2 = DoubleConv(128, 64)

            self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.conv3 = DoubleConv(64, 32)

            self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
            self.sigmoid = nn.Sigmoid()

        def forward_logits(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)

            x = self.up1(x4)
            x = self.conv1(torch.cat([x, x3], dim=1))

            x = self.up2(x)
            x = self.conv2(torch.cat([x, x2], dim=1))

            x = self.up3(x)
            x = self.conv3(torch.cat([x, x1], dim=1))

            return self.outc(x)

        def forward(self, x):
            return self.sigmoid(self.forward_logits(x))
else:
    class RiceCanopyUNet:
        pass
