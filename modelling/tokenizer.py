# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
import os
from omegaconf import OmegaConf

from modelling.modules import Encoder,Decoder,TimmViTEncoder,TimmViTDecoder,TimmSemanticViTEncoder,TimmSemanticViTDecoder
from modelling.quantizers.vq import VectorQuantizer
from modelling.quantizers.kl import DiagonalGaussianDistribution
from modelling.quantizers.softvq import SoftVectorQuantizer
from modelling.quantizers.simvq import simVectorQuantizer

from timm import create_model
from vfmtok.tokenizer.vq_model import VQModel as VFMTokInternalModel, ModelArgs as VFMTokInternalArgs


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


@dataclass
class ModelArgs:
    image_size: int = 256
    base_image_size: int = 256
    
    codebook_size: int = 16384
    codebook_embed_dim: int = 12
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    quantizer_type: str = "vq"
    vq_loss_ratio: float = 1.0 # for soft vq
    kl_loss_weight: float = 0.000001
    tau: float = 0.1
    num_codebooks: int = 1
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 512 # todo : add to yaml & config
    dropout_p: float = 0.0

    enc_type: str = 'cnn'
    dec_type: str = 'cnn'
    encoder_model: str = 'llamagen_encoder'
    decoder_model: str = 'llamagen_decoder'
    num_latent_tokens: int = 256
    to_pixel: str = 'linear'
    
    # for pre-trained models
    enc_tuning_method: str = 'full'
    dec_tuning_method: str = 'full'
    enc_pretrained: bool = True
    dec_pretrained: bool = False 
    enc_local_ckpt: str = ''
    dec_local_ckpt: str = ''
    repa_local_ckpt: str = ''
    
    # for vit 
    enc_patch_size: int = 16
    dec_patch_size: int = 16
    enc_drop_path_rate: float = 0.0
    dec_drop_path_rate: float = 0.0
    
    # deocder cls token
    dec_cls_token: bool = True
    
    # rope
    use_ape: bool = True 
    use_rope: bool = False
    rope_mixed: bool = False
    rope_theta: float = 10.0
    
    # repa for vit
    repa: bool = False
    repa_patch_size: int = 16
    repa_model: str = 'vit_base_patch16_224'
    repa_proj_dim: int = 2048
    repa_loss_weight: float = 0.1
    repa_align: str = 'global'
    cls_recon: bool = False
    cls_recon_weight: float = 0.0
    semantic_target_layers: List[int] = field(default_factory=lambda: [5, 11, 17, 23])
    semantic_qam_layers: int = 1
    transformer_config: Optional[str] = None
    codebook_slots_embed_dim: int = 12
    
    vq_mean: float = 0.0
    vq_std: float = 1.0
    
    # encoder token drop for mask modeling
    enc_token_drop: float = 0.0
    enc_token_drop_max: float = 0.6

    # aux decoder model
    aux_dec_model: str = 'vit_tiny_patch14_dinov2_movq'
    aux_loss_mask: bool = False
    aux_dec_cls_token: bool = True
    aux_hog_dec: bool = True
    aux_dino_dec: bool = True
    aux_supcls_dec: bool = True
    aux_clip_dec: bool = True
    decoder_up_type: str = "CNN"
    

class VQModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs, 
                tags=["arxiv:2412.10958", "image-generation", "32 tokens", "SoftVQ-VAE"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__()
        self.config = config
        self.vq_mean = config.vq_mean
        self.vq_std = config.vq_std
        self.num_latent_tokens = config.num_latent_tokens
        self.codebook_embed_dim = config.codebook_embed_dim
        
        self.repa = config.repa
        self.repa_loss_weight = config.repa_loss_weight
        self.repa_align = config.repa_align
        self.cls_recon = bool(config.cls_recon and self.repa and config.enc_type == 'vit')
        self.cls_recon_weight = config.cls_recon_weight if self.cls_recon else 0.0
        self.repa_cls_target = None
        if config.repa and config.enc_type == 'vit':
            local_repa_ckpt = (config.repa_local_ckpt or "").strip()
            has_local_repa = bool(local_repa_ckpt)
            if has_local_repa and not os.path.isfile(local_repa_ckpt):
                raise FileNotFoundError(
                    f"[REPA] Specified local checkpoint '{local_repa_ckpt}' does not exist."
                )

            try:
                if has_local_repa:
                    self.repa_model = create_model(
                        config.repa_model,
                        pretrained=True,
                        img_size=config.image_size,
                        patch_size=config.repa_patch_size,
                        pretrained_cfg_overlay={'file': local_repa_ckpt}
                    )
                else:
                    self.repa_model = create_model(
                        config.repa_model,
                        pretrained=True,
                        img_size=config.image_size,
                        patch_size=config.repa_patch_size
                    )
            except Exception as e:
                if has_local_repa:
                    self.repa_model = create_model(
                        config.repa_model,
                        pretrained=False,
                        img_size=config.image_size,
                        patch_size=config.repa_patch_size,
                        checkpoint_path=local_repa_ckpt
                    )
                else:
                    raise RuntimeError(
                        "[REPA] Failed to load pretrained weights from the internet. "
                        "Please provide 'repa_local_ckpt' in the config or set --repa-local-ckpt."
                    ) from e
            for param in self.repa_model.parameters():
                param.requires_grad = False
            self.repa_model.eval()
            repa_z_dim = self.repa_model.embed_dim
            self.repa_z_dim = repa_z_dim
            self.projection = build_mlp(config.codebook_embed_dim, config.repa_proj_dim, repa_z_dim)
            from modelling.lpips.lpips_timm import Normalize, Denormalize
            self.de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            repa_z_dim = None
        
        
        if config.enc_type == 'cnn':
            if config.encoder_model == 'llamagen_encoder':
                self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            else:
                raise NotImplementedError
            self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        elif config.enc_type == 'vit':
            self.encoder = TimmViTEncoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.encoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  #
                model_kwargs={'img_size': config.image_size, 'patch_size': config.enc_patch_size, 'drop_path_rate': config.enc_drop_path_rate},
                pretrained=config.enc_pretrained,
                tuning_method=config.enc_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                token_drop=config.enc_token_drop,
                token_drop_max=config.enc_token_drop_max,
                base_img_size=config.base_image_size,
                local_ckpt=config.enc_local_ckpt
            )
            self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim)
            
        if self.cls_recon:
            self.cls_prefix_projector = nn.Linear(self.repa_z_dim, self.encoder.embed_dim)
            if config.dec_type != 'vit':
                raise ValueError("CLS reconstruction requires a ViT decoder.")
        else:
            self.cls_prefix_projector = None
        self.cls_decoder_projector = None
        self.dec_cls_pred = None
        
        
        if config.dec_type == 'cnn':
            if config.decoder_model == 'llamagen_decoder':
                self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            else:
                raise NotImplementedError
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)
        elif config.dec_type == 'vit':
            kwargs = {'img_size': config.image_size, 'patch_size': config.dec_patch_size, 'drop_path_rate': config.dec_drop_path_rate, 'latent_dim': config.codebook_embed_dim}
            if 'dinov3' in config.decoder_model:
                kwargs.pop('latent_dim')
            
            self.decoder = TimmViTDecoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.decoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  
                model_kwargs=kwargs,
                pretrained=config.dec_pretrained,
                tuning_method=config.dec_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                cls_token=config.dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_pixel=config.to_pixel,
                base_img_size=config.base_image_size,
                local_ckpt=config.dec_local_ckpt
            )
            self.post_quant_conv = nn.Linear(config.codebook_embed_dim, self.decoder.embed_dim)
        # check movq
        if 'movq' in config.decoder_model:
            self.use_movq = True 
        else:
            self.use_movq = False

        if self.cls_recon:
            if getattr(self.decoder, 'num_prefix_tokens', 0) == 0:
                raise ValueError("Decoder does not expose prefix tokens; cannot enable CLS reconstruction.")
            self.cls_decoder_projector = nn.Linear(self.decoder.embed_dim, self.repa_z_dim)
        
        
        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage)

    def encode(self, x):
        repa_tokens = None
        encoder_prefix = None

        if self.repa:
            with torch.no_grad():
                rescale_x = self.scale(self.de_scale(x))
                repa_feats = self.repa_model.forward_features(rescale_x)
            repa_tokens = repa_feats[:, self.repa_model.num_prefix_tokens:]
            if self.cls_recon:
                if self.repa_model.num_prefix_tokens < 1:
                    raise ValueError("CLS reconstruction enabled but REPA model lacks prefix tokens.")
                repa_cls = repa_feats[:, :1, :].detach()
                self.repa_cls_target = repa_cls
                encoder_prefix = self.cls_prefix_projector(repa_cls)
                encoder_prefix = encoder_prefix.to(dtype=x.dtype, device=x.device)
                if self.encoder.num_prefix_tokens < 1:
                    raise ValueError("Encoder has no prefix tokens to receive CLS projection.")
            else:
                self.repa_cls_target = None
        else:
            self.repa_cls_target = None

        h = self.encoder(x, external_prefix=encoder_prefix)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        
        if self.repa and self.training:
            if repa_tokens is None:
                raise RuntimeError("REPA spatial tokens missing while REPA is enabled.")
            z = repa_tokens

            # taking average over spatial dimension
            if self.repa_align == 'global':
                z = z.mean(dim=1)
                z_hat = quant.mean(dim=1)
                # calculate repa loss
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d':
                z = F.adaptive_avg_pool1d(z.permute(0, 2, 1), quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d_shuffle':
                # shuffle the length dimension of z and avg
                indices = torch.randperm(z.shape[1])
                z = F.adaptive_avg_pool1d(z[:, indices, :].permute(0, 2, 1) , quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'repeat':
                z_hat = self.projection(quant)
                b, l, d = z_hat.shape
                z_hat = z_hat.unsqueeze(2).expand(-1, -1, z.size(1) // l, -1).reshape(b, -1, d)
            

            z = F.normalize(z, dim=-1)
            z_hat = F.normalize(z_hat, dim=-1)
            proj_loss = mean_flat(-(z * z_hat).sum(dim=-1))
            proj_loss = proj_loss.mean()
            proj_loss *= self.repa_loss_weight
            
            emb_loss += (proj_loss,)
        
        return quant, emb_loss, info

    def decode(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        cls_tokens = None
        if self.use_movq:
            if self.cls_recon:
                dec, cls_tokens = self.decoder(quant, tmp_quant, h, w, return_prefix=True)
            else:
                dec = self.decoder(quant, tmp_quant, h, w, return_prefix=False)
        else:
            if self.cls_recon:
                dec, cls_tokens = self.decoder(quant, None, h, w, return_prefix=True)
            else:
                dec = self.decoder(quant, None, h, w, return_prefix=False)
        if self.cls_recon:
            self.dec_cls_pred = cls_tokens[:, :1, :] if cls_tokens is not None else None
        else:
            self.dec_cls_pred = None
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        b, _, h, w = input.size()
        quant, diff, info = self.encode(input)
        self.quant = quant
        dec = self.decode(quant, x=input, h=h, w=w)
        
        if self.cls_recon and self.training:
            cls_loss = None
            if self.repa_cls_target is not None and self.dec_cls_pred is not None:
                pred = self.cls_decoder_projector(self.dec_cls_pred.to(dec.dtype))
                target = self.repa_cls_target.to(pred.device, dtype=pred.dtype)
                if pred.shape != target.shape:
                    raise ValueError(f"CLS prediction shape {pred.shape} does not match target shape {target.shape}.")
                mse = F.mse_loss(pred, target)
                pred_norm = F.normalize(pred, dim=-1)
                target_norm = F.normalize(target, dim=-1)
                cos_loss = mean_flat(-(pred_norm * target_norm).sum(dim=-1))
                cos_loss = cos_loss.mean()
                mse = self.cls_recon_weight * mse
                cos_loss = self.cls_recon_weight * cos_loss
                cls_loss = (mse, cos_loss)
            if cls_loss is not None:
                diff = diff + cls_loss
        return dec, diff, info


class SemanticVQModel(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        config: ModelArgs,
        tags=["semantic", "vit", "vq-vae"],
    ):
        super().__init__()
        self.config = config
        self.vq_mean = config.vq_mean
        self.vq_std = config.vq_std
        self.num_latent_tokens = config.num_latent_tokens
        self.codebook_embed_dim = config.codebook_embed_dim
        self.repa = config.repa
        self.repa_loss_weight = config.repa_loss_weight
        self.repa_align = config.repa_align

        self.encoder = TimmSemanticViTEncoder(
            model_name=config.encoder_model,
            model_kwargs={'img_size': config.image_size,'patch_size': config.enc_patch_size,'drop_path_rate': config.enc_drop_path_rate},
            pretrained=config.enc_pretrained,
            tuning_method=config.enc_tuning_method,
            tuning_kwargs={'r': 8},
            use_ape=config.use_ape,
            target_layers=config.semantic_target_layers,
            local_ckpt=config.enc_local_ckpt,
            num_latent_tokens=config.num_latent_tokens,
            qam_num_layers=config.semantic_qam_layers,
        )

        self.decoder = TimmSemanticViTDecoder(
            in_channels=3,
            model_name=config.decoder_model,
            model_kwargs={'img_size': config.image_size,'patch_size': config.dec_patch_size,'drop_path_rate': config.dec_drop_path_rate,'latent_dim': config.codebook_embed_dim,},
            pretrained=config.dec_pretrained,
            tuning_method=config.dec_tuning_method,
            tuning_kwargs={'r': 8},
            cls_token=config.dec_cls_token,
            codebook_embed_dim=config.codebook_embed_dim,
            to_pixel=config.to_pixel,
            base_img_size=config.base_image_size,
            local_ckpt=config.dec_local_ckpt,
            num_latent_tokens=config.num_latent_tokens,
            use_ape=config.use_ape,
            qam_num_layers=config.semantic_qam_layers,
        )

        self.quant_conv = nn.Linear(self.encoder.model.embed_dim, config.codebook_embed_dim)
        self.post_quant_conv = nn.Linear(config.codebook_embed_dim, self.decoder.embed_dim)

        self.quantize = simVectorQuantizer(
            codebook_size=config.codebook_size,
            token_size=config.codebook_embed_dim,
            commitment_cost=config.commit_loss_beta,
            use_l2_norm=config.codebook_l2_norm,
            simvq=True,
            entropy_loss_ratio=config.entropy_loss_ratio,
        )

        if self.repa:
            repa_dim = self.encoder.model.embed_dim
            self.repa_projection = build_mlp(
                config.codebook_embed_dim,
                config.repa_proj_dim,
                repa_dim,
            )
        else:
            self.repa_projection = None

        self.encoder_cache = {}

    def encode(self, x: torch.Tensor):
        latent_tokens, enc_dict = self.encoder(x)
        if latent_tokens is None:
            raise RuntimeError("Semantic ViT encoder must return latent tokens.")
        self.encoder_cache = enc_dict
        h = self.quant_conv(latent_tokens)
        quant, emb_loss, info = self.quantize(h)

        if self.repa and self.training:
            repa_feats = self.encoder_cache.get('image_features',None)
            if repa_feats is None:
                raise RuntimeError("Semantic encoder did not cache image_features for REPA loss.")

            prefix_tokens = getattr(self.encoder, "num_prefix_tokens", 0)
            repa_tokens = repa_feats[:, prefix_tokens:, :] if prefix_tokens > 0 else repa_feats
            z = repa_tokens

            if self.repa_align == 'global':
                z = z.mean(dim=1)
                z_hat = quant.mean(dim=1)
                z_hat = self.repa_projection(z_hat)
            elif self.repa_align == 'avg_1d':
                z = F.adaptive_avg_pool1d(z.permute(0, 2, 1), quant.shape[1]).permute(0, 2, 1)
                z_hat = self.repa_projection(quant)
            elif self.repa_align == 'avg_1d_shuffle':
                indices = torch.randperm(z.shape[1], device=z.device)
                z = F.adaptive_avg_pool1d(z[:, indices, :].permute(0, 2, 1), quant.shape[1]).permute(0, 2, 1)
                z_hat = self.repa_projection(quant)
            elif self.repa_align == 'repeat':
                z_hat = self.repa_projection(quant)
                b, l, d = z_hat.shape
                repeat_factor = z.size(1) // l
                z_hat = z_hat.unsqueeze(2).expand(-1, -1, repeat_factor, -1).reshape(b, -1, d)
            else:
                raise ValueError(f"Unsupported REPA alignment mode '{self.repa_align}'.")

            z = F.normalize(z, dim=-1)
            z_hat = F.normalize(z_hat, dim=-1)
            proj_loss = mean_flat(-(z * z_hat).sum(dim=-1))
            proj_loss = proj_loss.mean()
            proj_loss *= self.repa_loss_weight
            emb_loss += (proj_loss,)

        return quant, emb_loss, info

    def decode(self, quant: torch.Tensor, h: Optional[int] = None, w: Optional[int] = None):
        latent = self.post_quant_conv(quant)
        dec = self.decoder(latent, H=h, W=w)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input: torch.Tensor):
        b, _, h, w = input.size()
        quant, diff, info = self.encode(input)
        self.quant = quant
        dec = self.decode(quant, h=h, w=w)
        return dec, diff, info

    @torch.no_grad()
    def encode_to_ids(self, images: torch.Tensor, as_list: bool = False):
        was_training = self.training
        if was_training:
            self.eval()
        _, _, info = self.encode(images)
        indices = info[2].squeeze(1).contiguous()
        if was_training:
            self.train()
        if as_list:
            return indices.tolist()
        return indices

    @torch.no_grad()
    def decode_from_ids(self, indices: torch.Tensor, shape=None, channel_first=True):
        device = self.quantize.embedding.weight.device
        indices = indices.to(device)
        return self.decode_code(indices, shape=shape, channel_first=channel_first)


class VFMTokModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        transformer_cfg = config.transformer_config
        if transformer_cfg is None:
            raise ValueError("VFMTokModel requires 'transformer_config' in ModelArgs.")
        if isinstance(transformer_cfg, str):
            if not os.path.isfile(transformer_cfg):
                raise FileNotFoundError(f"Transformer config not found at '{transformer_cfg}'.")
            transformer_cfg = OmegaConf.load(transformer_cfg)

        internal_args = VFMTokInternalArgs(
            codebook_size=config.codebook_size,
            codebook_embed_dim=config.codebook_embed_dim,
            codebook_l2_norm=config.codebook_l2_norm,
            codebook_show_usage=config.codebook_show_usage,
            commit_loss_beta=config.commit_loss_beta,
            entropy_loss_ratio=config.entropy_loss_ratio,
            quantizer_type=config.quantizer_type,
            encoder_ch_mult=config.encoder_ch_mult,
            decoder_ch_mult=config.decoder_ch_mult,
            z_channels=config.z_channels,
            dropout_p=config.dropout_p,
            transformer_config=transformer_cfg,
            codebook_slots_embed_dim=config.codebook_slots_embed_dim,
            image_size=config.image_size,
            patch_size=config.dec_patch_size,
            in_channels=3,
            decoder_up_type=config.decoder_up_type,
        )

        self.model = VFMTokInternalModel(internal_args)
        self.decoder = self.model.decoder
        self.quantize = self.model.slot_quantize

    def encode(self, x: torch.Tensor):
        (quant_slots, latent), emb_loss, q_indices = self.model.encode(x)
        info = (
            latent,
            None,
            q_indices.view(x.size(0), 1, -1),
        )
        return quant_slots, emb_loss, info

    def decode(self, quant: torch.Tensor, h: Optional[int] = None, w: Optional[int] = None):
        dec, _ = self.model.decode(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        if shape is None:
            raise ValueError("VFMTokModel.decode_code requires 'shape' specifying (B, tokens, dim).")
        dec, _ = self.model.decode_code(code_b, shape, channel_first)
        return dec

    def forward(self, input: torch.Tensor):
        (dec, latent, dinov2), diff, q_indices = self.model(input)
        if self.training and latent is not None and dinov2 is not None:
            sim_loss = 1 - F.cosine_similarity(dinov2, latent, dim=-1)
            sim_loss = sim_loss.mean()
            diff += (sim_loss,)
        info = (
            latent,
            dinov2,
            q_indices.view(input.size(0), 1, -1),
        )
        return dec, diff, info

    @torch.no_grad()
    def encode_to_ids(self, images: torch.Tensor, as_list: bool = False):
        was_training = self.training
        if was_training:
            self.eval()
        _, _, info = self.encode(images)
        indices = info[2].squeeze(1).contiguous()
        if was_training:
            self.train()
        if as_list:
            return indices.tolist()
        return indices

    @torch.no_grad()
    def decode_from_ids(self, indices: torch.Tensor, shape=None, channel_first=True):
        if indices.dim() == 3 and indices.size(1) == 1:
            indices = indices.squeeze(1)
        flat = indices.reshape(indices.size(0), -1)
        qz_shape = (flat.size(0), flat.size(1), self.config.codebook_slots_embed_dim)
        dec, _ = self.model.decode_from_indices(flat.view(-1), qz_shape, channel_first=False)
        return dec


class SoftVQModel(VQModel, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs, 
                tags=["arxiv:2412.10958", "image-generation", "32 tokens", "SoftVQ-VAE"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__(config)
        self.quantize = SoftVectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                            config.entropy_loss_ratio, 
                                            config.tau,                                   
                                            config.num_codebooks,
                                            config.codebook_l2_norm, config.codebook_show_usage)

class simVQModel(VQModel, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.quantize = simVectorQuantizer(codebook_size=config.codebook_size, token_size=config.codebook_embed_dim, commitment_cost=config.commit_loss_beta,
                                           use_l2_norm=config.codebook_l2_norm,simvq=True,
                                           entropy_loss_ratio=config.entropy_loss_ratio)

    @torch.no_grad()
    def encode_to_ids(self, images: torch.Tensor, as_list: bool = False):
        """Encode images to discrete code indices of shape [B, L]."""
        was_training = self.training
        if was_training:
            self.eval()
        _, _, info = self.encode(images)
        indices = info[2].squeeze(1).contiguous()
        if was_training:
            self.train()
        if as_list:
            return indices.tolist()
        return indices

    @torch.no_grad()
    def decode_from_ids(self, indices: torch.Tensor, shape=None, channel_first=True):
        """Decode images from discrete code indices shaped [B, L] (or compatible)."""
        device = self.quantize.embedding.weight.device
        indices = indices.to(device)
        return self.decode_code(indices, shape=shape, channel_first=channel_first)

class KLModel(VQModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.kl_loss_weight = config.kl_loss_weight
        self.quantize = None
        
        if config.enc_type == 'cnn':
            self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim * 2, 1)
        elif config.enc_type == 'vit':
            self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim * 2)
        

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        # quant, emb_loss, info = self.quantize(h)
        h_posterior = DiagonalGaussianDistribution(h)
        return h_posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode_code(self, posterior, shape=None):
        z = posterior.sample()
        dec = self.decode(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        # compute kl loss
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        diff = (kl_loss * self.kl_loss_weight, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        return dec, diff, None


class AEModel(VQModel):
    def __init__(self, config: ModelArgs,
                tags=["arxiv:2502.03444", "image-generation", "1d-tokenizer", "128 tokens", "MAETok"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__(config)
        self.quantize = None 



    def encode(self, x):
        
        h = self.encoder(x)
        quant = self.quant_conv(h)
        emb_loss = (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = None
        return quant, emb_loss, info

    def decode(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        if self.use_movq:
            dec = self.decoder(quant, tmp_quant, h, w)
        else:
            dec = self.decoder(quant, None, h, w)
        return dec


class MaskAEModel(AEModel):
    def __init__(self, config: ModelArgs,
                tags=["arxiv:2502.03444", "image-generation", "1d-tokenizer", "128 tokens", "MAETok"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        config.repa = True # for dinov2 decoder
        super().__init__(config)

        self.aux_loss_mask = config.aux_loss_mask
        
        self.aux_hog_decoder = config.aux_hog_dec
        if self.aux_hog_decoder:
            from utils.hog import HOGGenerator
            print('Using HOG decoder: ', config.aux_dec_model)
            self.decoder_hog = TimmViTDecoder(
                in_channels=3, 
                num_latent_tokens=config.num_latent_tokens,
                model_name=config.aux_dec_model,
                model_kwargs={'img_size': config.image_size, 'patch_size': config.dec_patch_size, 'drop_path_rate': 0.0, 'latent_dim': config.codebook_embed_dim},
                pretrained=False,
                tuning_method='full',
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                cls_token=config.aux_dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_pixel='identity',
                base_img_size=config.base_image_size
            )
            self.post_quant_conv_hog = nn.Linear(config.codebook_embed_dim, self.decoder_hog.embed_dim)
            self.to_pixel_hog = nn.Linear(self.decoder_hog.embed_dim, 108)
            self.hog_generator = HOGGenerator()
            if 'movq' in config.aux_dec_model:
                self.hog_use_movq = True 
            else:
                self.hog_use_movq = False
        
        self.aux_dino_decoder = config.aux_dino_dec
        if self.aux_dino_decoder:
            print('Using DINO decoder: ', config.aux_dec_model)
            self.decoder_dino = TimmViTDecoder(
                in_channels=3, 
                num_latent_tokens=config.num_latent_tokens,
                model_name=config.aux_dec_model,
                model_kwargs={'img_size': self.repa_model.img_size, 'patch_size': self.repa_model.patch_size, 'drop_path_rate': 0.0, 'latent_dim': config.codebook_embed_dim},
                pretrained=False,
                tuning_method='full',
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                cls_token=config.aux_dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_pixel='identity',
                base_img_size=config.base_image_size
            )
            self.post_quant_conv_dino = nn.Linear(config.codebook_embed_dim, self.decoder_dino.embed_dim)
            self.to_pixel_dino = nn.Linear(self.decoder_dino.embed_dim, self.repa_model.embed_dim)
            if 'movq' in config.aux_dec_model:
                self.dino_use_movq = True 
            else:
                self.dino_use_movq = False
        
        self.aux_clip_decoder = config.aux_clip_dec
        if self.aux_clip_decoder:
            self.clip_model = create_model('vit_so400m_patch14_siglip_gap_224', pretrained=True, img_size=config.image_size, patch_size=config.repa_patch_size)
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.dynamic_img_size = True
            self.clip_model.eval()
            from modelling.lpips.lpips_timm import Normalize, Denormalize
            self.clip_de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.clip_scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            print('Using CLIP decoder: ', config.aux_dec_model)
            self.decoder_clip = TimmViTDecoder(
                in_channels=3, 
                num_latent_tokens=config.num_latent_tokens,
                model_name=config.aux_dec_model,
                model_kwargs={'img_size': self.clip_model.img_size, 'patch_size': self.clip_model.patch_size, 'drop_path_rate': 0.0, 'latent_dim': config.codebook_embed_dim},
                pretrained=False,
                tuning_method='full',
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                cls_token=config.aux_dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_pixel='identity',
                base_img_size=config.base_image_size
            )
            self.post_quant_conv_clip = nn.Linear(config.codebook_embed_dim, self.decoder_clip.embed_dim)
            self.to_pixel_clip = nn.Linear(self.decoder_clip.embed_dim, self.clip_model.embed_dim)
            if 'movq' in config.aux_dec_model:
                self.clip_use_movq = True 
            else:
                self.clip_use_movq = False


    def encode(self, x):
        
        h = self.encoder(x)
        if self.training:
            h, mask = self.encoder(x, return_mask=True)
        else:
            h = self.encoder(x)
        quant = self.quant_conv(h)
        emb_loss = (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = None
        if self.training:
            return quant, emb_loss, info, mask
        else:
            return quant, emb_loss, info

    def decode_hog(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv_hog(quant)
        if self.hog_use_movq:
            dec = self.decoder_hog(quant, tmp_quant, h, w)
        else:
            dec = self.decoder_hog(quant, None, h, w)
        return dec
    
    def decode_dino(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv_dino(quant)
        if self.dino_use_movq:
            dec = self.decoder_dino(quant, tmp_quant, h, w)
        else:
            dec = self.decoder_dino(quant, None, h, w)
        return dec

    def decode_clip(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv_clip(quant)
        if self.clip_use_movq:
            dec = self.decoder_clip(quant, tmp_quant, h, w)
        else:
            dec = self.decoder_clip(quant, None, h, w)
        return dec

    def forward(self, input):
        b, _, h, w = input.size()
        if self.training:
            quant, diff, info, mask = self.encode(input)
        else:
            quant, diff, info = self.encode(input)
        self.quant = quant
        # print(quant.shape)
        dec = self.decode(quant, x=input, h=h, w=w)
         
        # decode hog
        if self.training:
            # decode hog feature
            if self.aux_hog_decoder:
                dec_hog = self.decode_hog(quant, x=input, h=h, w=w)   
                dec_hog = self.to_pixel_hog(dec_hog)
                # get hog_target
                z_hog = self.hog_generator(input) 
                if self.aux_loss_mask:
                    hog_rec_loss = F.mse_loss(dec_hog, z_hog, reduction='none')
                    hog_rec_loss = (hog_rec_loss * mask).sum() / mask.sum() / z_hog.size(-1)
                else:
                    hog_rec_loss = F.mse_loss(dec_hog, z_hog)
            else:
                hog_rec_loss = 0.0
        
            # decode dinov2 feature
            if self.aux_dino_decoder:
                dec_dino = self.decode_dino(quant, x=input, h=h, w=w)
                dec_dino = self.to_pixel_dino(dec_dino)
                
                # get z from repa_encoder
                rescale_x = self.scale(self.de_scale(input))
                z_dino = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]

                z_dino = F.normalize(z_dino, dim=-1)
                dec_dino = F.normalize(dec_dino, dim=-1)

                if self.aux_loss_mask:
                    dino_rec_loss = -(dec_dino * z_dino).sum(dim=-1, keepdim=True)
                    dino_rec_loss = (dino_rec_loss * mask).sum() / mask.sum()
                else:
                    dino_rec_loss = mean_flat(-(dec_dino * z_dino).sum(dim=-1))
                    dino_rec_loss = dino_rec_loss.mean()
            else:
                dino_rec_loss = 0.0
            
            # deocde clip feature
            if self.aux_clip_decoder:
                dec_clip = self.decode_clip(quant, x=input, h=h, w=w)
                dec_clip = self.to_pixel_clip(dec_clip)
                # get clip_target
                rescale_x = self.clip_scale(self.clip_de_scale(input))
                z_clip = self.clip_model.forward_features(rescale_x)[:, self.clip_model.num_prefix_tokens:]
                
                z_clip = F.normalize(z_clip, dim=-1)
                dec_clip = F.normalize(dec_clip, dim=-1)
                
                if self.aux_loss_mask:
                    clip_rec_loss = -(dec_clip * z_clip).sum(dim=-1, keepdim=True)
                    clip_rec_loss = (clip_rec_loss * mask).sum() / mask.sum()
                else:
                    clip_rec_loss = mean_flat(-(dec_clip * z_clip).sum(dim=-1))
                    clip_rec_loss = clip_rec_loss.mean()   
            else:
                clip_rec_loss = 0.0
            
            diff += (dino_rec_loss, hog_rec_loss, clip_rec_loss, )

        return dec, diff, info

#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def VQ_16(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def KL_8(**kwargs):
    return KLModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def KL_16(**kwargs):
    return KLModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def AE_16(**kwargs):
    return AEModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def MaskAE_16(**kwargs):
    return MaskAEModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def SoftVQ(**kwargs):
    return SoftVQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def simVQ(**kwargs):
    return simVQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def SemanticVQ(**kwargs):
    return SemanticVQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def VFMTok(**kwargs):
    return VFMTokModel(ModelArgs(**kwargs))

VQ_models = {
    'AE-16': AE_16,
    'MaskAE-16': MaskAE_16,
    'VQ-16': VQ_16, 'VQ-8': VQ_8,
    'KL-16': KL_16, 'KL-8': KL_8,
    'SoftVQ': SoftVQ,
    'simVQ': simVQ,
    'SemanticVQ': SemanticVQ,
    'VFMTok': VFMTok,
}


if __name__ == '__main__':
    
    # model = VQ_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', repa=True, repa_model='vit_base_patch14_dinov2.lvd142m', repa_align='avg_1d_shuffle', enc_img_res=True, enc_img_align='avg_1d', dec_img_res=True)    
    # model = SoftVQ_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', num_codebooks=4, codebook_size=16384, topk=16)
    # model = AE_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', num_codebooks=4, codebook_size=16384)
    model = MaskAE_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', num_codebooks=4, codebook_size=16384, aux_dec_model='vit_tinytiny_patch14_dinov2_movq2', aux_loss_mask=True, aux_dec_cls_token=True, aux_hog_dec=True, aux_dino_dec=True, aux_clip_dec=True, enc_token_drop=0.4, enc_token_drop_max=0.6)      
    model.train()
    model = model.cuda()
    # model = KL_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m')
    # model = GMM_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m')
    x = torch.randn(4, 3, 256, 256).cuda()
    y, _, info = model(x)
    print(_)
