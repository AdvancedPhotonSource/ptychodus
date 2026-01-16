"""
CNN Decoders for SYNAPS-I reconstruction models.
"""

import torch
import torch.nn as nn


class TransposeConvBlock(nn.Module):
    """Transposed convolution block with optional batch norm and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        *,
        activation: str | None = 'relu',
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation is not None:
            raise ValueError(f'Unsupported activation: {activation}')

        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomActivation(nn.Module):
    """Custom activation used by the SYNAPS-I decoders."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.7159 * torch.tanh((2 / 3) * x)


class Decoder256(nn.Module):
    """CNN Decoder for 256x256 images with configurable depth."""

    def __init__(
        self,
        *,
        latent_dim: int = 512,
        base_channels: int = 64,
        out_channels: int = 1,
        use_batchnorm: bool = True,
        output_activation: str | None = None,
        dropout: float = 0.0,
        num_stages: int = 4,
    ) -> None:
        super().__init__()

        self.num_stages = num_stages
        self.stages = nn.ModuleList()

        encoder_channels = []
        for index in range(num_stages):
            if index < num_stages - 1:
                channel = min(base_channels * (2**index), latent_dim)
            else:
                channel = latent_dim
            encoder_channels.append(channel)

        in_channels = latent_dim
        for index in range(num_stages):
            if index < num_stages - 1:
                out_ch = encoder_channels[num_stages - 2 - index]
            else:
                out_ch = base_channels

            stage = nn.Sequential(
                TransposeConvBlock(
                    in_channels,
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                ),
                TransposeConvBlock(
                    out_ch,
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                ),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            )
            self.stages.append(stage)
            in_channels = out_ch

        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1, stride=1, padding=0)

        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        elif output_activation == 'custom':
            self.output_activation = CustomActivation()
        elif output_activation is None:
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f'Unsupported output activation: {output_activation}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        x = self.output(x)
        return self.output_activation(x)
