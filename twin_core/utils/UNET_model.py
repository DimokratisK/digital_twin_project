# utils/UNET_model.py
import torch
import torch.nn as nn
from typing import Optional


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True, dropout: Optional[float] = None):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        if dropout is not None and dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        super().__init__(*layers)


class UNet(nn.Module):
    """
    Simple 2D UNet.

    - in_channels: input image channels (1 for grayscale)
    - out_channels: number of output channels (logits). For multi-class segmentation use n_classes.
    - base_features: number of features in the first encoder stage (commonly 32 or 64)
    - use_bn: whether to use BatchNorm
    - dropout: optional dropout probability applied after each block
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 64,
        use_bn: bool = True,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        f = base_features
        # Encoder
        self.enc1 = ConvBlock(in_channels, f, use_bn=use_bn, dropout=dropout)
        self.enc2 = ConvBlock(f, f * 2, use_bn=use_bn, dropout=dropout)
        self.enc3 = ConvBlock(f * 2, f * 4, use_bn=use_bn, dropout=dropout)
        self.enc4 = ConvBlock(f * 4, f * 8, use_bn=use_bn, dropout=dropout)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16, use_bn=use_bn, dropout=dropout)

        # Decoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8, use_bn=use_bn, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4, use_bn=use_bn, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2, use_bn=use_bn, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2, f, use_bn=use_bn, dropout=dropout)

        # Final 1x1 conv to produce logits
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)            # (B, f, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 2f, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 4f, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 8f, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 16f, H/16, W/16)

        # Decoder with skip connections
        d4 = self.up4(b)                       # (B, 8f, H/8, W/8)
        d4 = torch.cat([d4, e4], dim=1)        # (B, 16f, H/8, W/8)
        d4 = self.dec4(d4)                     # (B, 8f, H/8, W/8)

        d3 = self.up3(d4)                      # (B, 4f, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)        # (B, 8f, H/4, W/4)
        d3 = self.dec3(d3)                     # (B, 4f, H/4, W/4)

        d2 = self.up2(d3)                      # (B, 2f, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)        # (B, 4f, H/2, W/2)
        d2 = self.dec2(d2)                     # (B, 2f, H/2, W/2)

        d1 = self.up1(d2)                      # (B, f, H, W)
        d1 = torch.cat([d1, e1], dim=1)        # (B, 2f, H, W)
        d1 = self.dec1(d1)                     # (B, f, H, W)

        logits = self.out_conv(d1)             # (B, out_channels, H, W)
        return logits

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
