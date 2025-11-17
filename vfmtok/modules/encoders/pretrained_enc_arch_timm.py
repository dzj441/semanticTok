import torch
import torch.nn as nn
import timm
from typing import Sequence, Optional
from omegaconf import OmegaConf
from timm.data import resolve_model_data_config
from vfmtok.engine.util import disabled_train
import kornia
import os


class TimmEmbedder(nn.Module):
    """
    SigLIP v2 embedder that returns multi-level token features (NLC) for specified transformer blocks.
    - Freezes the vision backbone.
    - Normalizes inputs using timm data_config.
    - Returns a list of [B, L, C] tensors (one per requested level), with prefix tokens removed.
    """
    def __init__(
        self,
        model_name: str = "vit_large_patch16_siglip_384.v2_webli",
        pretrained: bool = True,
        feature_levels: Sequence[int] = (5, 11, 17, 23),
        img_size: int = 384,
        patch_size: int = 16,
        ckpt_path: Optional[str] = "./weights/vit_large_patch16_siglip_384.v2_webli/model.safetensors",
        antialias: bool = True,
    ):
        super().__init__()
        # build backbone, optionally load local checkpoint
        pretrained_cfg_overlay = None
        if ckpt_path:
            pretrained_cfg_overlay = {"file": ckpt_path}
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            patch_size=patch_size,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.train = disabled_train

        # basic info
        self.feature_levels = list(feature_levels)
        self.img_size = img_size
        self.antialias = antialias
        self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
        self.embed_dim = getattr(self.model, "embed_dim", None)

        # timm data config for mean/std/size
        cfg = resolve_model_data_config(self.model)
        mean = torch.tensor(cfg.get("mean", (0.5, 0.5, 0.5))).view(1, 3, 1, 1)
        std = torch.tensor(cfg.get("std", (0.5, 0.5, 0.5))).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    @torch.no_grad()
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # normalize to [0,1] and resize with kornia to align token grid (e.g., 384 @ ps=16 -> 24x24=576 tokens)
        x = kornia.geometry.resize(
            x,
            (384, 384),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = x.clamp(min=-1, max=1)
        x = (x + 1.0) / 2.0
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = self.preprocess(x)
        # Apply model-specific mean/std normalization after mapping to [0,1]
        x = (x - self.mean) / self.std.to(x.device)

        # Request NLC intermediates at desired block indices (prenorm by default)
        feats = self.model.forward_intermediates(
            x, indices=self.feature_levels,
            return_prefix_tokens=False, norm=False, stop_early=False,
            output_fmt="NLC", intermediates_only=True
        )
        # apply final LayerNorm only to the last level (force apply)
        if len(feats) > 0:
            feats[-1] = self.model.norm(feats[-1])

        return feats
