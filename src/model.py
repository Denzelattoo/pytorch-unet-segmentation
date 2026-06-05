import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), 

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)  
        )

    def forward(self, x):
        return self.conv_block(x)


class Unet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x = self.pool1(x1)

        x2 = self.down2(x)
        x = self.pool2(x2)

        x3 = self.down3(x)
        x = self.pool3(x3)

        x_bottom = self.bottleneck(x)

        up3 = self.up3(x_bottom)
        cat3 = torch.cat([up3, x3], dim=1)
        x = self.conv_up3(cat3)

        up2 = self.up2(x)
        cat2 = torch.cat([up2, x2], dim=1)
        x = self.conv_up2(cat2)

        up1 = self.up1(x)
        cat1 = torch.cat([up1, x1], dim=1)
        x = self.conv_up1(cat1)

        output = self.final_conv(x)
        return output