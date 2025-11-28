import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialChannelTradeOffBlock(nn.Module):
    """
    Downsample spatial resolution, optional channel splitting, and optional pixel unshuffle.
    """

    def __init__(
        self,
        token_length: int,
        embed_dim: int,
        factor: int = 2,
        method: str = "interp",
        split_channels: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.factor = factor
        self.method = method
        self.split_channels = split_channels
        self.channel_factor = factor * factor

        h = int(math.sqrt(token_length))
        if h * h != token_length:
            raise ValueError("token_length must form a square grid.")
        self.input_hw = (h, h)
        self.low_hw = (h // factor, h // factor)

        self.pixel_unshuffle = nn.PixelUnshuffle(factor) if method == "pixel" and factor > 1 else None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        h, w = self.input_hw

        x = tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        if self.factor == 1:
            out = x
        elif self.method == "pixel":
            out = self.pixel_unshuffle(x)
        else:
            out = F.interpolate(x, size=self.low_hw, mode="bicubic", align_corners=False)

        total_c = self.embed_dim * self.channel_factor if (self.method == "pixel" and self.factor > 1) else self.embed_dim
        out = out.permute(0, 2, 3, 1).reshape(b, -1, total_c)

        if self.split_channels and self.channel_factor > 1:
            new_c = total_c // self.channel_factor
            out = out.view(b, -1, self.channel_factor, new_c).reshape(b, -1, new_c)

        return out


class SpatialChannelTradeOffInverse(nn.Module):
    """
    Inverse of SpatialChannelTradeOffBlock.
    """

    def __init__(
        self,
        token_length: int,
        embed_dim: int,
        factor: int = 2,
        method: str = "interp",
        split_channels: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.factor = factor
        self.method = method
        self.split_channels = split_channels
        self.channel_factor = factor * factor

        h = int(math.sqrt(token_length))
        if h * h != token_length:
            raise ValueError("token_length must form a square grid.")
        self.output_hw = (h, h)
        self.low_hw = (h // factor, h // factor)

        self.pixel_shuffle = nn.PixelShuffle(factor) if method == "pixel" and factor > 1 else None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        low_len = self.low_hw[0] * self.low_hw[1]

        base_c = self.embed_dim * self.channel_factor if (self.method == "pixel" and self.factor > 1) else self.embed_dim
        if self.split_channels and self.channel_factor > 1:
            expected_c = base_c // self.channel_factor
            expected_len = low_len * self.channel_factor
        else:
            expected_c = base_c
            expected_len = low_len

        if c != expected_c or n != expected_len:
            raise ValueError("SpatialChannelTradeOffInverse received mismatched token shape.")

        if self.split_channels and self.channel_factor > 1:
            merged = tokens.view(b, low_len, self.channel_factor, c).reshape(b, low_len, base_c)
        else:
            merged = tokens.view(b, low_len, base_c)

        x = merged.view(b, self.low_hw[0], self.low_hw[1], base_c).permute(0, 3, 1, 2).contiguous()

        if self.factor == 1:
            out = x
        elif self.method == "pixel":
            out = self.pixel_shuffle(x)
        else:
            out = F.interpolate(x, size=self.output_hw, mode="bicubic", align_corners=False)

        out = out.permute(0, 2, 3, 1).reshape(b, -1, self.embed_dim)
        return out
