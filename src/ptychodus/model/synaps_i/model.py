"""
SYNAPS-I model definition copied from ptycho-vit.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .decoders import Decoder256
from .vit import CustomViT


class PtychoViT(nn.Module):
    """Vision Transformer-based ptychography reconstruction model."""

    def __init__(self, config: dict[str, object] | None = None) -> None:
        super().__init__()
        if config is None:
            config = {
                'encoder_type': 'custom',
                'encoder': {
                    'img_size': 256,
                    'patch_size': 16,
                    'in_channels': 1,
                    'embed_dim': 512,
                    'depth': 12,
                    'num_heads': 8,
                    'mlp_ratio': 4.0,
                    'use_cls_token': False,
                    'dropout': 0.1,
                    'attn_dropout': 0.0,
                },
                'decoder': {
                    'base_channels': 64,
                    'latent_dim': None,
                    'use_batchnorm': True,
                    'dropout': 0.1,
                    'num_stages': 4,
                },
            }

        encoder_type = str(config.get('encoder_type', 'custom'))
        encoder_config = dict(config.get('encoder', {}))

        if encoder_type == 'pretrained':
            raise ModuleNotFoundError(
                'Pretrained encoder requires timm; only custom encoder is supported in SYNAPS-I.'
            )

        self.encoder = CustomViT(
            img_size=int(encoder_config.get('img_size', 256)),
            patch_size=int(encoder_config.get('patch_size', 16)),
            in_channels=int(encoder_config.get('in_channels', 1)),
            embed_dim=int(encoder_config.get('embed_dim', 512)),
            depth=int(encoder_config.get('depth', 12)),
            num_heads=int(encoder_config.get('num_heads', 8)),
            mlp_ratio=float(encoder_config.get('mlp_ratio', 4.0)),
            dropout=float(encoder_config.get('dropout', 0.1)),
            attn_dropout=float(encoder_config.get('attn_dropout', 0.0)),
            use_cls_token=bool(encoder_config.get('use_cls_token', False)),
        )

        decoder_config = dict(config.get('decoder', {}))
        latent_dim = decoder_config.get('latent_dim')
        if latent_dim is None:
            latent_dim = self.encoder.embed_dim

        self.amp_decoder = Decoder256(
            latent_dim=int(latent_dim),
            base_channels=int(decoder_config.get('base_channels', 64)),
            out_channels=1,
            use_batchnorm=bool(decoder_config.get('use_batchnorm', True)),
            output_activation='custom',
            dropout=float(decoder_config.get('dropout', 0.1)),
            num_stages=int(decoder_config.get('num_stages', 4)),
        )
        self.ph_decoder = Decoder256(
            latent_dim=int(latent_dim),
            base_channels=int(decoder_config.get('base_channels', 64)),
            out_channels=1,
            use_batchnorm=bool(decoder_config.get('use_batchnorm', True)),
            output_activation='custom',
            dropout=float(decoder_config.get('dropout', 0.1)),
            num_stages=int(decoder_config.get('num_stages', 4)),
        )

        self.log_scale_amp = nn.Parameter(
            torch.tensor(math.log(0.025), dtype=torch.float32), requires_grad=False
        )
        self.log_scale_ph = nn.Parameter(
            torch.tensor(math.log(math.pi), dtype=torch.float32), requires_grad=False
        )

    def forward(
        self,
        x: torch.Tensor,
        probe: torch.Tensor,
        normalization: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = 2 * torch.log10(x + 1e-1)

        probe = torch.complex(probe[:, :, :, :, :, 0], probe[:, :, :, :, :, 1])

        normalization = normalization.view(normalization.shape[0], 1, 1)
        scale = scale.view(scale.shape[0], 1, 1)

        x = self.encoder(x)

        constrained_amp = self.amp_decoder(x).squeeze(1)
        constrained_ph = self.ph_decoder(x).squeeze(1)

        amp = (constrained_amp * torch.exp(self.log_scale_amp)) + 0.975
        ph = constrained_ph * torch.exp(self.log_scale_ph)

        complex_object = torch.complex(amp * torch.cos(ph), amp * torch.sin(ph))
        psi = torch.fft.fftshift(
            torch.fft.fft2(complex_object[:, None, None, :] * probe), dim=(-2, -1)
        )
        intensity = (psi.abs() ** 2).sum(2)[:, 0]
        intensity = (intensity.float() / normalization) * scale

        pred_diff_amp = torch.sqrt(intensity)
        return pred_diff_amp.unsqueeze(1), amp.unsqueeze(1), ph.unsqueeze(1)
