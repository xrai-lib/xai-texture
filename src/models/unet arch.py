import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def _init_(self, n_channels, n_classes):
        super(UNet, self)._init_()
        # Encoder
        self.conv1 = self.contract_block(n_channels, 64)
        self.conv2 = self.contract_block(64, 128)
        self.conv3 = self.contract_block(128, 256)
        self.conv4 = self.contract_block(256, 512)
        self.conv5 = self.contract_block(512, 1024)

        # Decoder
        self.upconv5 = self.expand_block(1024, 512)
        self.upconv4 = self.expand_block(512, 256)
        self.upconv3 = self.expand_block(256, 128)
        self.upconv2 = self.expand_block(128, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def _forward_(self, x):
        # Encoder
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        # Decoder
        u4 = self.upconv5(c5, c4)
        u3 = self.upconv4(u4, c3)
        u2 = self.upconv3(u3, c2)
        u1 = self.upconv2(u2, c1)

        # Final layer
        out = self.final_conv(u1)
        return out

    def contract_block(self, in_channels, out_channels, kernel_size=3):
        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size=3):
        expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        return expand

    def forward(self, x):
        # Encode
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        # Decode
        u4 = self.upconv5(c5, c4)
        u3 = self.upconv4(u4, c3)
        u2 = self.upconv3(u3, c2)
        u1 = self.upconv2(u2, c1)

        # Output layer
        out = self.final_conv(u1)
        return out

# Example
# model = UNet(n_channels=3, n_classes=1)
# print(model)