import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import partial
import scipy.stats as stats
import os
import peft
from typing import Any, Dict, List, Optional
from timm.models import create_model
from timm.layers import trunc_normal_, resample_abs_pos_embed
from modelling.modules.timm_vit.to_pixel import ToPixel
from modelling.modules.timm_vit.vision_transformer import Attention, MoVQNorm, MoVQBlockv2
from modelling.modules.queryAttention import QueryAttnModule
from modelling.modules.timm_vit.rope_utils import compute_axial_cis, compute_mixed_cis, init_random_2d_freqs, init_t_xy
def wrap_eva_block(block, num_prefix_tokens, num_latent_tokens):
    if getattr(block, "_rope_wrapped", False):
        return

    original_attn_forward = block.attn.forward

    def patched_attn_forward(x, rope=None, attn_mask=None, **kwargs):
        if rope is not None:
            rope_is_tensor = torch.is_tensor(rope)
            sin = cos = None
            if rope_is_tensor:
                try:
                    sin, cos = rope.chunk(2, dim=-1)
                except RuntimeError:
                    sin = cos = None
            elif isinstance(rope, (tuple, list)) and len(rope) == 2:
                cos, sin = rope

            if sin is not None and cos is not None:
                patch_len = cos.shape[-2]
                total_tokens = x.shape[1]
                latent_tokens = max(total_tokens - num_prefix_tokens - patch_len, 0)
                if latent_tokens > 0:
                    pad_shape = (*cos.shape[:-2], latent_tokens, cos.shape[-1])
                    cos_pad = cos.new_ones(pad_shape)
                    sin_pad = sin.new_zeros(pad_shape)
                    cos = torch.cat([cos, cos_pad], dim=-2)
                    sin = torch.cat([sin, sin_pad], dim=-2)

                rope = torch.cat([sin, cos], dim=-1) if rope_is_tensor else (cos, sin)

        return original_attn_forward(x, rope=rope, attn_mask=attn_mask, **kwargs)

    block.attn.forward = patched_attn_forward
    block._rope_wrapped = True

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


class TimmViTEncoder(nn.Module):
    def __init__(self, in_channels=3, num_latent_tokens=32,
                 model_name='vit_small_patch14_dinov2.lvd142m',
                 model_kwargs={'img_size': 224, 'patch_size': 14, 'drop_path_rate': 0.0,},
                 pretrained=True, tuning_method='full', tuning_kwargs={'r': 8},
                 rope_theta=100.0, rope_mixed=False, use_rope=False, use_ape=True,
                 token_drop=0.4,
                 token_drop_max=0.6,
                 base_img_size=224,
                 local_ckpt: str = ''
                 ):
        super().__init__()

        self.model_name = model_name
        assert model_name in ['vit_small_patch14_dinov2.lvd142m', 'vit_base_patch14_dinov2.lvd142m',
                              'vit_large_patch14_dinov2.lvd142m', 'vit_giant_patch14_dinov2.lvd142m',
                              'vit_small_patch14_reg4_dinov2.lvd142m', 'vit_base_patch14_reg4_dinov2.lvd142m',
                              'vit_large_patch14_reg4_dinov2.lvd142m', 
                              'vit_large_patch16_dinov3.lvd1689m', 'vit_base_patch16_dinov3.lvd1689m', 
                              'vit_giant_patch14_reg4_dinov2.lvd142m', 'vit_base_patch16_clip_224.openai',
                              "vit_base_patch16_clip_224.laion2b", "samvit_base_patch16.sa1b", "eva02_base_patch16_clip_224.merged2b"], f"{model_name} not found"

        # parameters
        self.num_latent_tokens = num_latent_tokens

        local_vit_ckpt = (local_ckpt or "").strip()
        has_local_ckpt = bool(local_vit_ckpt)
        if has_local_ckpt and not os.path.isfile(local_vit_ckpt):
            raise FileNotFoundError(
                f"[Encoder] Specified local checkpoint '{local_vit_ckpt}' does not exist."
            )

        if pretrained:
            if has_local_ckpt:
                model = create_model(
                    model_name,
                    pretrained=True,
                    pretrained_cfg_overlay={"file": local_vit_ckpt},
                    **model_kwargs,
                )
            else:
                model = create_model(
                    model_name,
                    pretrained=True,
                    **model_kwargs,
                )
        else:
            model = create_model(
                    model_name,
                    pretrained=False,
                    **model_kwargs,
                )

        self.img_size = model_kwargs['img_size']
        self.model_name = model_name
        self.is_dinov3 = 'dinov3' in model_name.lower()

        self.patch_size = model_kwargs['patch_size']
        self.embed_dim = model.embed_dim
        # get num of img tokens
        self.num_img_tokens = model.patch_embed.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        
        # tuning method
        if tuning_method == 'full':
            # doing nothing
            self.model = model
        elif tuning_method == 'lora':
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d",
                                     modules_to_save=['norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False
            self.model = model

        if self.num_latent_tokens:
            # latent tokens
            self.latent_tokens = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
            nn.init.normal_(self.latent_tokens, std=.02)

            self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
            trunc_normal_(self.latent_pos_embed, std=.02)

        # token drop
        self.token_drop = token_drop > 0.0
        if self.token_drop:
            # self.mask_ratio_generator = stats.truncnorm((1.0 - token_drop) / 0.25, 1.0 / 0.25, loc=1.0, scale=0.25)
            self.mask_ratio_generator = stats.truncnorm((token_drop - token_drop_max) / 0.25, 0, loc=token_drop_max, scale=0.25)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, model.embed_dim))
            nn.init.normal_(self.mask_token, std=.02)   
        # rope
        self.use_ape = use_ape
        self.use_rope = use_rope
        if self.is_dinov3:
            self.use_ape = False
            self.use_rope = False
        elif self.use_rope:
            self.use_ape = False
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        if not self.is_dinov3:
            if self.rope_mixed and self.use_rope:
                self.compute_cis = partial(compute_mixed_cis, num_heads=model.num_heads)
                
                freqs = []
                for i, _ in enumerate(model.blocks):
                    freqs.append(
                        init_random_2d_freqs(dim=model.embed_dim // model.num_heads, num_heads=model.num_heads, theta=self.rope_theta)
                    )
                freqs = torch.stack(freqs, dim=1).view(2, len(model.blocks), -1)
                self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
                
                if base_img_size != model_kwargs['img_size']:
                    t_x, t_y = init_t_xy(end_x = base_img_size // model_kwargs['patch_size'] , end_y =  base_img_size //  model_kwargs['patch_size'] )
                else:
                    t_x, t_y = init_t_xy(end_x = model_kwargs['img_size'] // model_kwargs['patch_size'] , end_y =  model_kwargs['img_size'] //  model_kwargs['patch_size'] )
                self.register_buffer('freqs_t_x', t_x)
                self.register_buffer('freqs_t_y', t_y)
            else:
                self.compute_cis = partial(compute_axial_cis, dim=model.embed_dim//model.num_heads, theta=rope_theta)
                
                freqs_cis = self.compute_cis(end_x = model_kwargs['img_size'] // model_kwargs['patch_size'] , end_y = model_kwargs['img_size'] //  model_kwargs['patch_size'] )
                self.freqs_cis = freqs_cis

        if not self.is_dinov3 and not self.use_ape:
            for b in self.model.blocks:
                b.attn.flash_attn = False
        if self.is_dinov3:
            for blk in self.model.blocks:
                wrap_eva_block(blk, self.num_prefix_tokens, self.num_latent_tokens)


    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'latent_tokens', 'latent_pos_embed', 'freqs']

    def sample_orders(self, bsz, seq_len):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward(self, x, return_mask=False, external_prefix=None):

        # get tokens
        _, _, H, W = x.shape
        x = self.model.patch_embed(x)

        if self.token_drop and self.training:
            orders = self.sample_orders(bsz=x.size(0), seq_len=x.size(1)).to(x.device)
            mask = self.random_masking(x, orders).unsqueeze(-1)
            x = torch.where(mask.bool(), self.mask_token, x)
        else:
            mask = None 
        B = x.shape[0]
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size

        rope = None
        pos_embed = None
        if self.is_dinov3:
            if x.dim() == 3:
                x = x.view(B, grid_h, grid_w, -1)
            pos_out = self.model._pos_embed(x)
            if isinstance(pos_out, tuple):
                x, rope = pos_out[0], pos_out[1]
            else:
                x = pos_out
            if x.dim() == 4:
                x = x.view(B, -1, x.shape[-1])
        else:
            if getattr(self.model, 'pos_embed', None) is not None:
                num_prefix = 0 if getattr(self.model, 'no_embed_class', False) else self.num_prefix_tokens
                pos_embed = resample_abs_pos_embed(
                    self.model.pos_embed,
                    (grid_h, grid_w),
                    num_prefix_tokens=num_prefix,
                )
                pos_embed = pos_embed.to(dtype=x.dtype, device=x.device)

            if x.dim() == 4:
                x = x.view(B, -1, x.shape[-1])

        if self.is_dinov3: # dinov3
            tokens = x
            if external_prefix is not None:
                ext = external_prefix
                if ext.dim() == 2:
                    ext = ext.unsqueeze(1)
                if ext.shape[0] != B:
                    ext = ext.expand(B, -1, -1)
                ext = ext.to(dtype=tokens.dtype, device=tokens.device)
                if ext.shape[1] == 1 and self.num_prefix_tokens >= 1:
                    prefix_slice = tokens[:, :self.num_prefix_tokens, :]
                    remainder = prefix_slice[:, 1:, :] if self.num_prefix_tokens > 1 else None
                    new_prefix = ext if remainder is None else torch.cat([ext, remainder], dim=1)
                else:
                    if ext.shape[1] != self.num_prefix_tokens:
                        raise ValueError(f"Expected {self.num_prefix_tokens} prefix tokens, got {ext.shape[1]}.")
                    new_prefix = ext
                tokens = torch.cat([new_prefix, tokens[:, self.num_prefix_tokens:, :]], dim=1)
        else:
            if external_prefix is not None:  # external only pass the first token, i.e., cls token
                if external_prefix.dim() == 2:
                    external_prefix = external_prefix.unsqueeze(1)
                if external_prefix.shape[0] != B:
                    external_prefix = external_prefix.expand(B, -1, -1)

                base_tokens = []
                if getattr(self.model, 'cls_token', None) is not None:
                    base_tokens.append(self.model.cls_token.expand(B, -1, -1).to(x.dtype))
                if getattr(self.model, 'reg_token', None) is not None:
                    base_tokens.append(self.model.reg_token.expand(B, -1, -1).to(x.dtype))
                base_tokens = torch.cat(base_tokens, dim=1) if base_tokens else None

                if base_tokens is None:
                    prefix_tokens = external_prefix.to(dtype=x.dtype, device=x.device)
                elif external_prefix.shape[1] == 1 and base_tokens.shape[1] >= 1:
                    cls_token = external_prefix.to(dtype=x.dtype, device=x.device)
                    remaining = base_tokens[:, 1:, :] if base_tokens.shape[1] > 1 else None
                    prefix_tokens = cls_token if remaining is None else torch.cat([cls_token, remaining], dim=1)
                else:
                    prefix_tokens = external_prefix.to(dtype=x.dtype, device=x.device)
            else:
                to_cat = []
                if getattr(self.model, 'cls_token', None) is not None:
                    to_cat.append(self.model.cls_token.expand(B, -1, -1).to(x.dtype))
                if getattr(self.model, 'reg_token', None) is not None:
                    to_cat.append(self.model.reg_token.expand(B, -1, -1).to(x.dtype))
                prefix_tokens = torch.cat(to_cat, dim=1) if len(to_cat) > 0 else None

            if prefix_tokens is not None and prefix_tokens.shape[1] != self.num_prefix_tokens:
                raise ValueError(f"Expected {self.num_prefix_tokens} prefix tokens, got {prefix_tokens.shape[1]}.")

            if getattr(self.model, 'no_embed_class', False):
                tokens = x
                if self.use_ape and pos_embed is not None:
                    tokens = tokens + pos_embed
                if prefix_tokens is not None:
                    tokens = torch.cat([prefix_tokens, tokens], dim=1)
            else:
                if prefix_tokens is not None:
                    tokens = torch.cat([prefix_tokens, x], dim=1)
                else:
                    tokens = x
                if self.use_ape and pos_embed is not None:
                    tokens = tokens + pos_embed

        expected_patch_tokens = grid_h * grid_w
        actual_patch_tokens = tokens.shape[1] - self.num_prefix_tokens
        if actual_patch_tokens != expected_patch_tokens:
            raise AssertionError(
                f"[Encoder] Patch token count mismatch before pos_drop. "
                f"expected={expected_patch_tokens}, got={actual_patch_tokens}"
            )

        x = self.model.pos_drop(tokens)
        if hasattr(self.model, 'patch_drop') and self.model.patch_drop is not None:
            x = self.model.patch_drop(x)
        tokens_before_latent = x.shape[1]
        if self.num_latent_tokens:
            # insert latent tokens
            z = self.latent_tokens.expand(x.size(0), -1, -1)
            x = torch.cat([x, z + self.latent_pos_embed], dim=1)
        expected_total_tokens = self.num_prefix_tokens + expected_patch_tokens + self.num_latent_tokens
        if tokens_before_latent != self.num_prefix_tokens + expected_patch_tokens:
            raise AssertionError(
                f"[Encoder] Token length mismatch prior to latent insertion. "
                f"expected={self.num_prefix_tokens + expected_patch_tokens}, got={tokens_before_latent}"
            )
        if x.shape[1] != expected_total_tokens:
            raise AssertionError(
                f"[Encoder] Token length mismatch before blocks. "
                f"expected={expected_total_tokens}, got={x.shape[1]} "
                f"(prefix={self.num_prefix_tokens}, patch={expected_patch_tokens}, latent={self.num_latent_tokens})"
            )

        # pre layer norm
        if not 'eva02' in self.model_name:
            x = self.model.norm_pre(x)

        if self.is_dinov3 and rope is not None:
            for blk in self.model.blocks:
                x = blk(x, rope=rope)
        elif self.use_ape: 
            for i, blk in enumerate(self.model.blocks):
                x = blk(x)
        elif self.rope_mixed and self.use_rope:
            if self.freqs_t_x.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                t_x, t_y = init_t_xy(end_x = W // self.patch_size, end_y = H // self.patch_size)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            
            for i , blk in enumerate(self.model.blocks):
                x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
        else:
            if self.freqs_cis.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                freqs_cis = self.compute_cis(end_x = W // self.patch_size, end_y = H // self.patch_size)
            else:
                freqs_cis = self.freqs_cis
            freqs_cis = freqs_cis.to(x.device)
            
            for i , blk in enumerate(self.model.blocks):
                x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                
        # x = self.model.blocks(x)
        if not 'eva02' in self.model_name:
            x = self.model.norm(x)
        else:
            x = self.model.fc_norm(x)

        if self.num_latent_tokens:
            # get z tokens as out
            out = x[:, -self.num_latent_tokens:]
        else:
            # get img tokens as out
            out = x[:, self.num_prefix_tokens:]
        
        if return_mask:
            return out, mask
        else:
            return out


class TimmViTDecoder(nn.Module):
    def __init__(self, in_channels=3,
                 model_name='vit_small_patch14_dinov2.lvd142m',
                 model_kwargs={'img_size': 224, 'patch_size': 14, 'drop_path_rate': 0.0}, pretrained=True,
                 tuning_method='lora', tuning_kwargs={'r': 8},
                 num_latent_tokens=32, to_pixel='linear',
                 codebook_embed_dim=32,
                 rope_theta=100.0, rope_mixed=False, use_rope=False, use_ape=True,
                 cls_token=True,
                 base_img_size=224,
                 local_ckpt: str = ''
                 ):
        super().__init__()

        # model_kwargs['num_latent_tokens'] = num_latent_tokens
        # model_kwargs['class_token'] = cls_token

        # load model
        local_vit_ckpt = (local_ckpt or "").strip()
        has_local_ckpt = bool(local_vit_ckpt)
        if has_local_ckpt and not os.path.isfile(local_vit_ckpt):
            raise FileNotFoundError(
                f"[Decoder] Specified local checkpoint '{local_vit_ckpt}' does not exist."
            )

        if pretrained:
            if has_local_ckpt:
                model = create_model(
                    model_name,
                    pretrained=True,
                    pretrained_cfg_overlay={"file": local_vit_ckpt},
                    **model_kwargs,
                )
            else:
                model = create_model(
                    model_name,
                    pretrained=True,
                    **model_kwargs,
                )
        else:
            model = create_model(
                    model_name,
                    pretrained=False,
                    **model_kwargs,
                )


        self.model_name = model_name
        self.is_dinov3 = 'dinov3' in model_name.lower()
        self.patch_size = model_kwargs['patch_size']
        self.embed_dim = model.embed_dim
        # get num of img tokens
        self.num_img_tokens = model.patch_embed.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        self.num_latent_tokens = num_latent_tokens
        
        # tuning method
        if tuning_method == 'full':
            # doing nothing
            self.model = model
        elif tuning_method == 'lora':
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d",
                                     modules_to_save=['patch_embed.proj', 'patch_embed.norm', 'norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False
            self.model = model

        # latent tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model.embed_dim))
        nn.init.normal_(self.mask_token, std=.02)

        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
        trunc_normal_(self.latent_pos_embed, std=.02)

        # to pixel
        self.to_pixel = ToPixel(to_pixel=to_pixel, img_size=model_kwargs['img_size'], in_channels=in_channels,
                                in_dim=model.embed_dim, patch_size=model_kwargs['patch_size'])

        
        self.use_ape = use_ape
        self.use_rope = use_rope
        if self.is_dinov3:
            self.use_ape = False
            self.use_rope = False
        elif self.use_rope:
            self.use_ape = False
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        if not self.is_dinov3:
            if self.rope_mixed and self.use_rope:
                self.compute_cis = partial(compute_mixed_cis, num_heads=model.num_heads)
                
                freqs = []
                for i, _ in enumerate(model.blocks):
                    freqs.append(
                        init_random_2d_freqs(dim=model.embed_dim // model.num_heads, num_heads=model.num_heads, theta=self.rope_theta)
                    )
                freqs = torch.stack(freqs, dim=1).view(2, len(model.blocks), -1)
                self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
                
                if base_img_size != model_kwargs['img_size']:
                    t_x, t_y = init_t_xy(end_x = base_img_size // model_kwargs['patch_size'] , end_y =  base_img_size //  model_kwargs['patch_size'] )
                else:
                    t_x, t_y = init_t_xy(end_x = model_kwargs['img_size'] // model_kwargs['patch_size'] , end_y =  model_kwargs['img_size'] //  model_kwargs['patch_size'] )
                self.register_buffer('freqs_t_x', t_x)
                self.register_buffer('freqs_t_y', t_y)
            elif not self.rope_mixed and self.use_rope:
                self.compute_cis = partial(compute_axial_cis, dim=model.embed_dim//model.num_heads, theta=rope_theta)

                freqs_cis = self.compute_cis(end_x = model_kwargs['img_size'] // model_kwargs['patch_size'] , end_y = model_kwargs['img_size'] //  model_kwargs['patch_size'] )
                self.freqs_cis = freqs_cis

        if not self.is_dinov3 and not self.use_ape:
            for b in self.model.blocks:
                b.attn.flash_attn = False
        if self.is_dinov3:
            for blk in self.model.blocks:
                wrap_eva_block(blk, self.num_prefix_tokens, self.num_latent_tokens)


        if 'movq' in model_name:
            self.use_movq = True 
            self.model.norm = MoVQNorm(codebook_embed_dim, model.embed_dim)

            # Zero-out adaLN modulation layers in DiT blocks:
            for block in self.model.blocks:
                if isinstance(block, MoVQBlockv2):
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            if isinstance(self.model.norm, MoVQNorm):
                nn.init.constant_(self.model.norm.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.model.norm.adaLN_modulation[-1].bias, 0)
        else:
            self.use_movq = False 
            

        self.cls_token = cls_token
        if not cls_token:
            self.model.cls_token = None
            self.num_prefix_tokens -= 1
            self.model.num_prefix_tokens -= 1
            
    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'mask_token', 'latent_pos_embed', 'freqs']

    @property
    def last_layer(self):
        return self.to_pixel.get_last_layer()


    def forward(self, z, interpolate_zq=None, H=None, W=None, return_prefix=False):

        if H is None:
            num_img_tokens = self.num_img_tokens
            H = W = int(math.sqrt(num_img_tokens)) * self.patch_size
        else:
            num_img_tokens = H * W // self.patch_size ** 2

        # mask tokens
        if self.num_latent_tokens:
            if H is None:
                x = self.mask_token.expand(z.size(0), num_img_tokens, -1)
            else:
                x = self.mask_token.expand(z.size(0), H * W // self.patch_size ** 2, -1)
        else:
            x = z 
        rope = None
        _pos_embed = getattr(self.model, '_pos_embed')
        if self.is_dinov3:
            x = x.view(z.size(0), H // self.patch_size, W // self.patch_size, -1)
            pos_out = _pos_embed(x)
            if isinstance(pos_out, tuple):
                x, rope = pos_out[0], pos_out[1]
            else:
                x = pos_out
            if x.dim() == 4:
                x = x.view(z.size(0), -1, x.shape[-1])
        else:
            try:
                x = _pos_embed(x, use_ape=self.use_ape)
            except TypeError:
                x = _pos_embed(x)
            if isinstance(x, tuple):
                x = x[0]

        actual_patch_tokens = x.shape[1] - self.num_prefix_tokens
        if actual_patch_tokens != num_img_tokens:
            raise AssertionError(
                f"[Decoder] Patch token count mismatch before latent concat. "
                f"expected={num_img_tokens}, got={actual_patch_tokens}"
            )

        if hasattr(self.model, 'patch_drop') and self.model.patch_drop is not None:
            x = self.model.patch_drop(x)

        z = z + self.latent_pos_embed
        tokens_before_latent = x.shape[1]
        x = torch.cat([x, z], dim=1)
        expected_total_tokens = tokens_before_latent + self.num_latent_tokens
        if x.shape[1] != expected_total_tokens:
            raise AssertionError(
                f"[Decoder] Token length mismatch after latent concat. "
                f"expected={expected_total_tokens}, got={x.shape[1]}"
            )

        x = self.model.norm_pre(x)
        
        if self.is_dinov3 and rope is not None:
            for blk in self.model.blocks:
                if self.use_movq:
                    x = blk(x, rope=rope, interpolate_zq=interpolate_zq, num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x, rope=rope)
        elif self.use_ape: 
            for i, blk in enumerate(self.model.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq=interpolate_zq, num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x)
                
        elif self.rope_mixed and self.use_rope:
            if self.freqs_t_x.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                t_x, t_y = init_t_xy(end_x = W // self.patch_size, end_y = H // self.patch_size)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            
            for i , blk in enumerate(self.model.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)

        else:
            if self.freqs_cis.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                freqs_cis = self.compute_cis(end_x = W // self.patch_size, end_y = H // self.patch_size)
            else:
                freqs_cis = self.freqs_cis
            freqs_cis = freqs_cis.to(x.device)
            
            for i , blk in enumerate(self.model.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq,  freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)      

        if self.use_movq:
            x = self.model.norm(x, interpolate_zq,  num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
        else:
            x = self.model.norm(x)

        prefix_tokens = None
        if self.num_prefix_tokens > 0:
            prefix_tokens = x[:, :self.num_prefix_tokens]
        x = x[:, self.num_prefix_tokens:self.num_img_tokens + self.num_prefix_tokens]

        out = self.to_pixel(x)

        if return_prefix:
            return out, prefix_tokens
        return out


class TimmSemanticViTEncoder(nn.Module):
    """Lightweight wrapper that surfaces intermediate ViT features  for DINO feature pyramids).
       returns a feature pyramids for dino feature
    """

    def __init__(
        self,
        model_name: str = 'vit_large_patch14_dinov2.lvd142m',
        model_kwargs: dict = None,
        pretrained: bool = True,
        tuning_method: str = 'frozen',
        tuning_kwargs: dict = None,
        use_ape: bool = True,
        target_layers: Optional[List[int]] = None,
        local_ckpt: str = '',
        num_latent_tokens: int = 128,
        qam_num_layers: int = 1,
    ):
        super().__init__()
        model_kwargs = model_kwargs or {'img_size': 224, 'patch_size': 14, 'drop_path_rate': 0.0}
        tuning_kwargs = tuning_kwargs or {}
        local_vit_ckpt = (local_ckpt or "").strip()
        has_local_ckpt = bool(local_vit_ckpt)
        if has_local_ckpt and not os.path.isfile(local_vit_ckpt):
            raise FileNotFoundError(
                f"[SemanticEncoder] Specified local checkpoint '{local_vit_ckpt}' does not exist."
            )

        if pretrained:
            if has_local_ckpt:
                model = create_model(
                    model_name,
                    pretrained=True,
                    pretrained_cfg_overlay={"file": local_vit_ckpt},
                    **model_kwargs,
                )
            else:
                model = create_model(
                    model_name,
                    pretrained=True,
                    **model_kwargs,
                )
        else:
                model = create_model(
                    model_name,
                    pretrained=False,
                    **model_kwargs,
                )


        if tuning_method == 'frozen':
            for p in model.parameters():
                p.requires_grad = False
        elif tuning_method == 'lora':
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d", modules_to_save=['norm'], **tuning_kwargs)
            model = peft.get_peft_model(model, config)
        # tuning_method == 'full' keeps everything trainable

        self.model = model
        self.use_ape = use_ape
        self.num_latent_tokens = int(num_latent_tokens)
        if self.num_latent_tokens > 0:
            self.latent_tokens = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
            nn.init.normal_(self.latent_tokens, std=.02)
            self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
            trunc_normal_(self.latent_pos_embed, std=.02)
        else:
            self.latent_tokens = None
            self.latent_pos_embed = None
        default_layers = [5, 11, 17, 23] if len(model.blocks) >= 24 else [len(model.blocks) - 1]
        target_layers = default_layers if target_layers is None else list(target_layers)
        self.target_layers = sorted(set(l for l in target_layers if 0 <= l < len(model.blocks)))
        if not self.target_layers:
            raise ValueError("target_layers must contain valid block indices.")
        self._target_layer_set = set(self.target_layers)
        self.num_prefix_tokens = getattr(self.model, 'num_prefix_tokens', 0)
        self.num_semantic_layers = len(self.target_layers)
        self.layer_embeddings = nn.Parameter(torch.zeros(1, self.num_semantic_layers, model.embed_dim))
        trunc_normal_(self.layer_embeddings, std=.02)
        if self.num_latent_tokens > 0:
            self.qam = QueryAttnModule(
                d_model=model.embed_dim,
                n_heads=model.num_heads,
                num_layers=qam_num_layers,
            )
        else:
            self.qam = None
        
        if not self.use_ape:
            for b in self.model.blocks:
                b.attn.flash_attn = False

    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'latent_tokens', 'latent_pos_embed', 'freqs']

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_prefix_tokens: bool = True,
        intermediates_only: bool = False,
        norm : bool = True,
    ) -> Dict[str, Any]:
        """Return intermediate ViT features with optional prefix tokens."""
        self.model.eval()
        outputs = []

        B, _, height, width = x.shape
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x) # cls tokne cat to patch token here
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)

        for idx, blk in enumerate(self.model.blocks):
            if attn_mask is not None:
                x = blk(x, attn_mask=attn_mask)
            else:
                x = blk(x)
            if idx in self._target_layer_set:
                outputs.append(self.model.norm(x) if norm else x) # dino feats are LayerNormed

        image_features = self.model.norm(x) # dino feat

        prefix_tokens = None
        if self.num_prefix_tokens and outputs:
            prefix_tokens = [feat[:, :self.num_prefix_tokens] for feat in outputs]
            spatial_feats = [feat[:, self.num_prefix_tokens:] for feat in outputs]
        else:
            spatial_feats = outputs
            prefix_tokens = None

        latent_out = None

        typed_feats = []
        for idx, feat in enumerate(spatial_feats):
            layer_embed = self.layer_embeddings[:, idx:idx+1, :]
            typed_feats.append(feat + layer_embed)
        spatial_concat = torch.cat(typed_feats, dim=1)
        if self.qam is not None:
            latent = self.latent_tokens.expand(B, -1, -1)
            latent = latent + self.latent_pos_embed
            latent_out, _ = self.qam(query = latent, context = spatial_concat)

        # return unprocessed dino features in dict
        result_dict = {'image_intermediates': spatial_feats}
        if not intermediates_only:
            result_dict['image_features'] = image_features
        if return_prefix_tokens and prefix_tokens is not None:
            prefix_concat = torch.cat(prefix_tokens, dim=1)
            result_dict['image_intermediates_prefix'] = prefix_concat

        return latent_out, result_dict

class TimmSemanticViTDecoder(nn.Module):
    def __init__(self, in_channels=3,
                 model_name='vit_large_patch14_dinov2.lvd142m',
                 model_kwargs={'img_size': 256, 'patch_size': 16, 'drop_path_rate': 0.0}, pretrained=True,
                 tuning_method='full', tuning_kwargs=None,
                 num_latent_tokens=128, to_pixel='linear',
                 codebook_embed_dim=64,
                 use_rope=False, use_ape=True,
                 cls_token=True,
                 base_img_size=224,
                 local_ckpt: str = '',
                 qam_num_layers: int = 1
                 ):
        super().__init__()

        # model_kwargs['num_latent_tokens'] = num_latent_tokens
        # model_kwargs['class_token'] = cls_token

        # load model
        local_vit_ckpt = (local_ckpt or "").strip()
        has_local_ckpt = bool(local_vit_ckpt)
        if has_local_ckpt and not os.path.isfile(local_vit_ckpt):
            raise FileNotFoundError(
                f"[Decoder] Specified local checkpoint '{local_vit_ckpt}' does not exist."
            )

        if pretrained:
            if has_local_ckpt:
                model = create_model(
                    model_name,
                    pretrained=True,
                    pretrained_cfg_overlay={"file": local_vit_ckpt},
                    **model_kwargs,
                )
            else:
                model = create_model(
                    model_name,
                    pretrained=True,
                    **model_kwargs,
                )
        else:
            model = create_model(
                    model_name,
                    pretrained=False,
                    **model_kwargs,
                )

        self.model_name = model_name
        self.patch_size = model_kwargs['patch_size']
        self.embed_dim = model.embed_dim
        # get num of img tokens
        self.num_img_tokens = model.patch_embed.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        self.num_latent_tokens = num_latent_tokens
        
        # tuning method
        if tuning_method == 'full':
            # doing nothing
            self.model = model
        elif tuning_method == 'lora':
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d",
                                     modules_to_save=['patch_embed.proj', 'patch_embed.norm', 'norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False
            self.model = model

        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model.embed_dim))
        nn.init.normal_(self.mask_token, std=.02)

        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
        trunc_normal_(self.latent_pos_embed, std=.02)

        # to pixel
        self.to_pixel = ToPixel(to_pixel=to_pixel, img_size=model_kwargs['img_size'], in_channels=in_channels,
                                in_dim=model.embed_dim, patch_size=model_kwargs['patch_size'])
        self.use_ape = use_ape
        self.use_rope = use_rope
        if self.use_rope:
            self.use_ape = False

        if not self.use_ape:
            for b in self.model.blocks:
                b.attn.flash_attn = False

        if 'movq' in model_name:
            self.use_movq = True 
            self.model.norm = MoVQNorm(codebook_embed_dim, model.embed_dim)

            # Zero-out adaLN modulation layers in DiT blocks:
            for block in self.model.blocks:
                if isinstance(block, MoVQBlockv2):
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            if isinstance(self.model.norm, MoVQNorm):
                nn.init.constant_(self.model.norm.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.model.norm.adaLN_modulation[-1].bias, 0)
        else:
            self.use_movq = False 
            
        if self.num_latent_tokens > 0:
            self.qam = QueryAttnModule(
                d_model=self.model.embed_dim,
                n_heads=self.model.num_heads,
                num_layers=qam_num_layers,
            )
        else:
            self.qam = None

        self.cls_token = cls_token
        if not cls_token:
            self.model.cls_token = None
            self.num_prefix_tokens -= 1
            self.model.num_prefix_tokens -= 1
            
    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'mask_token', 'latent_pos_embed', 'freqs']

    @property
    def last_layer(self):
        return self.to_pixel.get_last_layer()


    def forward(self, z, interpolate_zq=None, H=None, W=None):

        if H is None:
            num_img_tokens = self.num_img_tokens
            H = W = int(math.sqrt(num_img_tokens)) * self.patch_size
        else:
            num_img_tokens = H * W // self.patch_size ** 2

        # mask tokens
        if self.num_latent_tokens:
            if H is None:
                x = self.mask_token.expand(z.size(0), num_img_tokens, -1)
            else:
                x = self.mask_token.expand(z.size(0), H * W // self.patch_size ** 2, -1)
        else:
            x = z 

        x = self.model._pos_embed(x, use_ape=self.use_ape) # cls concatenated

        actual_patch_tokens = x.shape[1] - self.num_prefix_tokens
        if actual_patch_tokens != num_img_tokens:
            raise AssertionError(
                f"[Decoder] Patch token count mismatch before latent concat. "
                f"expected={num_img_tokens}, got={actual_patch_tokens}"
            ) # shape check

        if self.num_prefix_tokens > 0:
            prefix_tokens = x[:, :self.num_prefix_tokens]
            mask_tokens = x[:, self.num_prefix_tokens:]
        else:
            prefix_tokens = None
            mask_tokens = x


        latent_tokens = z + self.latent_pos_embed

        if self.qam is not None:
            mask_tokens, _ = self.qam(query=mask_tokens, context=latent_tokens)
        if prefix_tokens is not None:
            x = torch.cat([prefix_tokens, mask_tokens], dim=1)
        else:
            x = mask_tokens

        if hasattr(self.model, 'patch_drop') and self.model.patch_drop is not None:
            x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        
        if self.use_ape: 
            for i, blk in enumerate(self.model.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq=interpolate_zq, num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x)
                
        if self.use_movq:
            x = self.model.norm(x, interpolate_zq,  num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
        else:
            x = self.model.norm(x)

        prefix_tokens = None
        if self.num_prefix_tokens > 0:
            prefix_tokens = x[:, :self.num_prefix_tokens]
        x = x[:, self.num_prefix_tokens:self.num_img_tokens + self.num_prefix_tokens]

        out = self.to_pixel(x)

        return out
if __name__ == '__main__':
    enc = TimmSemanticViTEncoder(model_kwargs={'img_size': 256, 'patch_size': 16},local_ckpt='./weights/vit_large_patch14_dinov2_lvd142m.pth')
    randnin = torch.randn(1,3,256,256)
    l,out_dict = enc.forward(randnin)
    for k,v in out_dict.items():
        print(k,v)
        print(v[-1])
    dec = TimmSemanticViTDecoder(model_kwargs={'img_size': 256, 'patch_size': 16},local_ckpt='./weights/vit_large_patch14_dinov2_lvd142m.pth')
    z = torch.randn(2,128,1024)
    dec_o = dec.forward(z,H=randnin.size(2),W=randnin.size(3))
    print(dec_o.shape)
