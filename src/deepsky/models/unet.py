"""DDPM U-Net: ResBlocks with time-embedding injection, self-attention at
configured resolutions, skip connections between down and up paths."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepsky.config import ModelCfg


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # Zero-init the last conv so each block starts as an identity + skip.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)
        out = (v @ attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, cfg: ModelCfg, image_size: int, in_channels: int = 3):
        super().__init__()
        self.cfg = cfg
        time_dim = cfg.time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim)
        )

        chans = [cfg.base_channels * m for m in cfg.channel_mults]
        self.stem = nn.Conv2d(in_channels, cfg.base_channels, 3, padding=1)

        # Down path. Track (channels, resolution) of every emitted feature map
        # so the up path can mirror them for skip connections.
        self.downs = nn.ModuleList()
        skip_chans = [cfg.base_channels]
        ch, res = cfg.base_channels, image_size
        for level, out_ch in enumerate(chans):
            for _ in range(cfg.num_res_blocks):
                block = nn.ModuleList(
                    [
                        ResBlock(ch, out_ch, time_dim, cfg.dropout),
                        AttentionBlock(out_ch) if res in cfg.attn_resolutions else nn.Identity(),
                    ]
                )
                self.downs.append(block)
                ch = out_ch
                skip_chans.append(ch)
            if level < len(chans) - 1:
                self.downs.append(Downsample(ch))
                res //= 2
                skip_chans.append(ch)

        self.mid1 = ResBlock(ch, ch, time_dim, cfg.dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid2 = ResBlock(ch, ch, time_dim, cfg.dropout)

        # Up path mirrors the down path, consuming skips in reverse.
        self.ups = nn.ModuleList()
        for level, out_ch in reversed(list(enumerate(chans))):
            for _ in range(cfg.num_res_blocks + 1):
                block = nn.ModuleList(
                    [
                        ResBlock(ch + skip_chans.pop(), out_ch, time_dim, cfg.dropout),
                        AttentionBlock(out_ch) if res in cfg.attn_resolutions else nn.Identity(),
                    ]
                )
                self.ups.append(block)
                ch = out_ch
            if level > 0:
                self.ups.append(Upsample(ch))
                res *= 2

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.cfg.time_emb_dim))

        h = self.stem(x)
        skips = [h]
        for module in self.downs:
            if isinstance(module, Downsample):
                h = module(h)
            else:
                res_block, attn = module
                h = attn(res_block(h, t_emb))
            skips.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), t_emb)

        for module in self.ups:
            if isinstance(module, Upsample):
                h = module(h)
            else:
                res_block, attn = module
                h = attn(res_block(torch.cat([h, skips.pop()], dim=1), t_emb))

        return self.out_conv(F.silu(self.out_norm(h)))
