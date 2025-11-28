# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
import torch, pdb
import torch.nn as nn
import os.path as osp
from typing import List, Optional
from copy import deepcopy
from einops import rearrange
import torch.nn.functional as F
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from ..engine.ema import requires_grad
from ..engine.util import instantiate_from_config

from einops.layers.torch import Rearrange
import numpy as np

class SineLayer(nn.Module):
    """
    Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN)
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class ToPixel(nn.Module):
    def __init__(self, to_pixel='linear', img_size=256, in_channels=3, in_dim=512, patch_size=16) -> None:
        super().__init__()
        self.to_pixel_name = to_pixel
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.in_channels = in_channels
        if to_pixel == 'linear':
            self.model = nn.Linear(in_dim, in_channels * patch_size * patch_size)
        elif to_pixel == 'conv':
            num_patches_per_dim = img_size // patch_size  # e.g. 256//16 = 16
            self.model = nn.Sequential(
                # (B, L, C) -> (B, C, H, W) with H = W = num_patches_per_dim
                Rearrange('b (h w) c -> b c h w', h=num_patches_per_dim),
                
                # For example, first reduce dimension via a 1x1 conv from in_dim -> 128
                nn.Conv2d(in_dim, 128, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),

                # Upsample from size (num_patches_per_dim) to a larger intermediate
                nn.Upsample(scale_factor=2, mode='nearest'),  
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                # Repeat upsampling until we reach the final resolution
                # For a 16x16 patch layout, we need 4x upsampling to reach 256
                #   16 -> 32 -> 64 -> 128 -> 256
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1),
            )
        elif to_pixel == 'siren':
            self.model = nn.Sequential(
                SineLayer(in_dim, in_dim * 2, is_first=True, omega_0=30.),
                SineLayer(in_dim * 2, img_size // patch_size * patch_size * in_channels, is_first=False, omega_0=30)
            )
        elif to_pixel == 'identity':
            self.model = nn.Identity()
        else:
            raise NotImplementedError

    def get_last_layer(self):
        if self.to_pixel_name == 'linear':
            return self.model.weight
        elif self.to_pixel_name == 'siren':
            return self.model[1].linear.weight
        elif self.to_pixel_name == 'conv':
            return self.model[-1].weight
        else:
            return None

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1], print(h, w, x.shape[1])
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, x):
        if self.to_pixel_name == 'linear':
            x = self.model(x)
            x = self.unpatchify(x)
        elif self.to_pixel_name == 'siren':
            x = self.model(x)
            x = x.view(x.shape[0], self.in_channels, self.patch_size * int(self.num_patches ** 0.5),
                       self.patch_size * int(self.num_patches ** 0.5))
        elif self.to_pixel_name == 'conv':
            x = self.model(x)
        elif self.to_pixel_name == 'identity':
            pass
        return x
    
@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 12
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    quantizer_type: str = "vq"
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 512
    dropout_p: float = 0.0
    transformer_config: str = None
    codebook_slots_embed_dim: int = 12
    image_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    decoder_up_type: str = "CNN" # todo : enable outer control of decoder type


class LinearPatchDecoder(nn.Module):
    """Wrap ToPixel-linear so downstream code can keep using `decoder.last_layer`."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, in_dim: int):
        super().__init__()
        self.pixel_head = ToPixel(
            to_pixel='linear',
            img_size=img_size,
            in_channels=in_channels,
            in_dim=in_dim,
            patch_size=patch_size,
        )

    @property
    def last_layer(self):
        return self.pixel_head.get_last_layer()

    def forward(self, z: torch.Tensor):
        if z.dim() == 4:
            z = rearrange(z, 'b c h w -> b (h w) c')
        return self.pixel_head(z)

class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.config = config
        if config.decoder_up_type == "linear":
            self.decoder = LinearPatchDecoder(
                img_size=config.image_size,
                patch_size=config.patch_size,
                in_channels=config.in_channels,
                in_dim=config.z_channels,
            )
        elif config.decoder_up_type == "semantic":
            self.decoder = nn.Identity() # identity don't need pixel decoder
        elif config.decoder_up_type == "CNN":
            self.decoder = Decoder(
                ch_mult=config.decoder_ch_mult,
                z_channels=config.z_channels,
                dropout=config.dropout_p,
            )

        self.encode_transformer = instantiate_from_config(config.transformer_config.encoder_config)
        self.decode_transformer = instantiate_from_config(config.transformer_config.decoder_config)
        self.tradeoff_block = (
            instantiate_from_config(config.transformer_config.tradeoff_block_config)
            if config.transformer_config.tradeoff_block_config is not None
            else None
        )
        self.inverse_tradeoff_block = (
            instantiate_from_config(config.transformer_config.inverse_tradeoff_block_config)
            if config.transformer_config.inverse_tradeoff_block_config is not None
            else None
        )
        if (self.tradeoff_block is None) ^ (self.inverse_tradeoff_block is None):
            raise ValueError("tradeoff_block_config and inverse_tradeoff_block_config must be both set or both None.")

        quantizer_type = getattr(config, "quantizer_type", "vq").lower()
        if quantizer_type == "vq":
            quantizer_cls = VectorQuantizer
        elif quantizer_type == "simvq":
            quantizer_cls = SimVectorQuantizer
        else:
            raise ValueError(f"Unsupported quantizer_type '{config.quantizer_type}'")

        self.slot_quantize = quantizer_cls(
            config.codebook_size,
            config.codebook_slots_embed_dim,
            config.commit_loss_beta,
            config.entropy_loss_ratio,
            config.codebook_l2_norm,
            config.codebook_show_usage,
        )
        self.pre_slots_quant = nn.Linear(config.z_channels, config.codebook_slots_embed_dim)
        self.post_slots_quant = nn.Linear(config.codebook_slots_embed_dim, config.z_channels)
        
        self.freeze_visual_encoder()

    def freeze_visual_encoder(self,):

        requires_grad(self.encode_transformer.backbone, False)

    def freeze(self,):

        requires_grad(self, False)

    def encode(self, x):

        slots, latent = self.encode_transformer(x,)
        if self.tradeoff_block is not None:
            slots = self.tradeoff_block(slots)
        queries = self.pre_slots_quant(slots)
        queries = rearrange(queries.unsqueeze(2), 'b h w c -> b c h w')
        quant2, emb_loss, (_, _, q_indices) = self.slot_quantize(queries)
        quant2 = rearrange(quant2, 'b c h w -> b (h w) c')

        return (quant2, latent), emb_loss, q_indices

    def decode_from_indices(self, q_indices, z_shape=None, channel_first=False):

        slots = self.slot_quantize.get_codebook_entry(q_indices, z_shape, channel_first)
        
        dec, dinov2 = self.decode(slots)

        return dec, dinov2

    def decode_codes_to_img(self, codes, tgt_size):

        bs, num = codes.shape
        qz_shape = (bs, num, self.config.codebook_slots_embed_dim,)
        results, _ = self.decode_code(codes, qz_shape, False)
        if results.shape[-1] != tgt_size:
            results = F.interpolate(results, size=(tgt_size, tgt_size), mode="bicubic")
        imgs = results.detach() * 127.5 + 128
        imgs = torch.clamp(imgs, 0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        return imgs

    def decode_tokens(self, codes, tgt_size):

        bs, num = codes.shape
        qz_shape = (bs, num, self.config.codebook_slots_embed_dim,)
        results, _ = self.decode_code(codes, qz_shape, False)
        if results.shape[-1] != tgt_size:
            results = F.interpolate(results, size=(tgt_size, tgt_size), mode="bicubic")
        imgs = results.detach() * 127.5 + 128
        imgs = torch.clamp(imgs, 0, 255).to(torch.uint8)
        return imgs

    def decode(self, slots):
        
        queries = self.post_slots_quant(slots)
        if self.inverse_tradeoff_block is not None:
            queries = self.inverse_tradeoff_block(queries)
        z, dinov2 = self.decode_transformer(queries)
        dec = self.decoder(z)
        return dec, dinov2

    def decode_code(self, code_b, shape=None, channel_first=True):

        quant_b = self.slot_quantize.get_codebook_entry(code_b, shape, channel_first)
        dec, dinov2 = self.decode(quant_b)
        return dec, dinov2

    def forward(self, input):

        (quant, latent), diff, q_indices = self.encode(input)
        if self.training:
            dec, dinov2 = self.decode(quant)
        else:
            dec, dinov2 = self.decode_from_indices(q_indices, quant.shape)
        return (dec, latent, dinov2), diff, q_indices

class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, 
                 norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions-1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        
        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h



class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=1, norm_type="group",
                 dropout=0.0, resamp_with_conv=True, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch*ch_mult[self.num_resolutions-1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

       # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                # if i_level == self.num_resolutions - 1:
                #     attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight
    
    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536*8)))

    
    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2) 
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


class SimVectorQuantizer(nn.Module):
    """SimVQ variant that preserves the original VQ interface (B,C,H,W inputs)."""

    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        entropy_loss_ratio,
        l2_norm,
        show_usage,
        simvq=True,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.simvq = simvq

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.e_dim ** -0.5)
        if self.simvq:
            for p in self.embedding.parameters():
                p.requires_grad = False
            self.embedding_proj = nn.Linear(self.e_dim, self.e_dim)

        if self.show_usage:
            self.register_buffer("codebook_used", torch.zeros(65536*8, dtype=torch.long))

    def _project_weight(self):
        weight = self.embedding.weight
        if self.simvq:
            weight = self.embedding_proj(weight)
        return weight

    def forward(self, z):
        if z.dim() != 4:
            raise ValueError(f"SimVectorQuantizer expects 4D inputs [B,C,H,W], got {tuple(z.shape)}")

        z_bhwc = z.permute(0, 2, 3, 1).contiguous()
        b, h, w, c = z_bhwc.shape
        z_flat = z_bhwc.view(b, -1, c)
        weight = self._project_weight()

        if self.l2_norm:
            z_for_dist = F.normalize(z_flat, dim=-1, eps=1e-6)
            weight_for_dist = F.normalize(weight, dim=-1, eps=1e-6)
        else:
            z_for_dist = z_flat
            weight_for_dist = weight

        z2 = (z_for_dist ** 2).sum(dim=-1, keepdim=True)
        w2 = (weight_for_dist ** 2).sum(dim=-1).view(1, 1, -1)
        dot = torch.einsum('bld,kd->blk', z_for_dist, weight_for_dist)
        dists = z2 + w2 - 2 * dot

        indices = torch.argmin(dists, dim=-1)
        lookup_weight = weight_for_dist if self.l2_norm else weight
        q = F.embedding(indices, lookup_weight).view(b, h, w, c)

        if self.training:
            z_cmp = F.normalize(z_bhwc, dim=-1, eps=1e-6) if self.l2_norm else z_bhwc
            q_cmp = F.normalize(q, dim=-1, eps=1e-6) if self.l2_norm else q
            codebook_loss = ((q_cmp - z_cmp.detach()) ** 2).mean()
            commit_loss = self.beta * ((q_cmp.detach() - z_cmp) ** 2).mean()
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-dists)
        else:
            codebook_loss = commit_loss = entropy_loss = None

        z_q = z_bhwc + (q - z_bhwc).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if self.show_usage and self.training:
            flat_idx = indices.reshape(-1)
            cur_len = min(flat_idx.shape[0], self.codebook_used.shape[0])
            if cur_len > 0:
                self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
                self.codebook_used[-cur_len:] = flat_idx[:cur_len].detach()
            usage = len(torch.unique(self.codebook_used)) / self.n_e
        else:
            usage = torch.tensor(0.0, device=z.device)

        flat_indices = indices.reshape(-1)
        return z_q, (codebook_loss, commit_loss, entropy_loss, usage), (None, None, flat_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        weight = self._project_weight()
        if self.l2_norm:
            weight = F.normalize(weight, dim=-1, eps=1e-6)

        if indices.dim() == 3:
            b = indices.size(0)
            indices = indices.view(b, -1)
        elif indices.dim() == 2:
            pass
        elif indices.dim() == 1:
            pass
        else:
            raise ValueError(f"Unsupported indices shape: {tuple(indices.shape)}")

        flat = indices.reshape(-1).long()
        feats = F.embedding(flat, weight)

        if shape is None:
            return feats

        if channel_first:
            b, c, h, w = shape
            feats = feats.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        else:
            feats = feats.view(*shape)
        return feats

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss


#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def VQ_16(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

VQ_models = {'VQ-16': VQ_16, 'VQ-8': VQ_8}
