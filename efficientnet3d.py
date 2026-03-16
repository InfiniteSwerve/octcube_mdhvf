"""
EfficientNet3D-B0 implementation for OCT volumetric regression.

Reference: Koyama et al., "Automated learning of glaucomatous visual fields
from OCT images using a comprehensive, segmentation-free 3D CNN model"
(Sci Rep 15, 13395, 2025).

Architecture: EfficientNet-B0 adapted to 3D (all 2D ops -> 3D), with:
  - MBConv blocks with 3D depthwise separable convolutions
  - 3D squeeze-and-excitation
  - 3D adaptive average pooling
  - 30% dropout before output

Implemented from scratch to avoid external dependencies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── EfficientNet-B0 block configuration ─────────────────────────────
# Format: (expand_ratio, channels, repeats, stride, kernel_size)
# This is the standard B0 architecture from the original paper.
B0_BLOCKS = [
    # expand, out_ch, repeats, stride, kernel
    (1,  16, 1, 1, 3),
    (6,  24, 2, 2, 3),
    (6,  40, 2, 2, 5),
    (6,  80, 3, 2, 3),
    (6, 112, 3, 1, 5),
    (6, 192, 4, 2, 5),
    (6, 320, 1, 1, 3),
]


class SqueezeExcite3D(nn.Module):
    """3D Squeeze-and-Excitation block."""
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        squeezed = max(1, int(channels * se_ratio))
        self.fc1 = nn.Conv3d(channels, squeezed, 1)
        self.fc2 = nn.Conv3d(squeezed, channels, 1)

    def forward(self, x):
        scale = x.mean(dim=(2, 3, 4), keepdim=True)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class MBConv3D(nn.Module):
    """3D Mobile Inverted Bottleneck (MBConv) block."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.0,
    ):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)
        self.drop_connect_rate = drop_connect_rate
        pad = kernel_size // 2

        layers = []
        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm3d(mid_ch),
                nn.SiLU(inplace=True),
            ])

        # Depthwise conv
        layers.extend([
            nn.Conv3d(mid_ch, mid_ch, kernel_size, stride=stride,
                      padding=pad, groups=mid_ch, bias=False),
            nn.BatchNorm3d(mid_ch),
            nn.SiLU(inplace=True),
        ])
        self.conv = nn.Sequential(*layers)

        # Squeeze-and-excitation
        self.se = SqueezeExcite3D(mid_ch, se_ratio)

        # Pointwise projection
        self.project = nn.Sequential(
            nn.Conv3d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm3d(out_ch),
        )

    def _drop_connect(self, x):
        if not self.training or self.drop_connect_rate == 0:
            return x
        keep = 1 - self.drop_connect_rate
        mask = torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) < keep
        return x * mask / keep

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = self._drop_connect(out) + x
        return out


class EfficientNet3D(nn.Module):
    """EfficientNet-B0 adapted to 3D volumes.

    Args:
        in_channels: Number of input channels (1 for grayscale OCT).
        num_classes: Output dimension. 1 for scalar regression.
        dropout_rate: Dropout before final linear layer (paper uses 0.3).
        first_stride: Stride for the stem conv. Use 2 to save memory on
                      large inputs (224^2 x 128).
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        dropout_rate: float = 0.3,
        first_stride: int = 2,
        drop_connect_rate: float = 0.2,
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=first_stride, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
        )

        # Build MBConv blocks
        blocks = []
        in_ch = 32
        total_blocks = sum(r for _, _, r, _, _ in B0_BLOCKS)
        block_idx = 0
        for expand, out_ch, repeats, stride, kernel in B0_BLOCKS:
            for i in range(repeats):
                s = stride if i == 0 else 1
                dc_rate = drop_connect_rate * block_idx / total_blocks
                blocks.append(MBConv3D(in_ch, out_ch, expand, kernel, s,
                                       drop_connect_rate=dc_rate))
                in_ch = out_ch
                block_idx += 1
        self.blocks = nn.Sequential(*blocks)

        # Head
        self.head_conv = nn.Sequential(
            nn.Conv3d(in_ch, 1280, 1, bias=False),
            nn.BatchNorm3d(1280),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1280, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W) — e.g. (B, 1, 128, 224, 224)
        Returns:
            (B,) regression predictions
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.pool(x).flatten(1)  # (B, 1280)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)  # (B,)


def efficientnet3d_b0(**kwargs) -> EfficientNet3D:
    """Convenience constructor for EfficientNet3D-B0."""
    return EfficientNet3D(**kwargs)
