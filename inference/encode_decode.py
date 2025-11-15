import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from modelling.tokenizer import VQ_models
from train.train_tokenizer import build_parser
from utils.misc import load_model_state_dict


def parse_args():
    parser = build_parser()
    parser.add_argument("--weights", type=str, default="./experiments/tokenizer/exp001-simvq-bl-128-reg/checkpoints/0250000.pth", help="Path to model-only weights (.pth).")
    parser.add_argument("--image", type=str, default="ImageNet/val/n01514668/ILSVRC2012_val_00000329.JPEG", help="Path to input image.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="encode_decode.png", help="Where to save the reconstruction.")
    args = parser.parse_args()
    if args.config and os.path.isfile(args.config):
        from ruamel import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**(config_args or {}))
        args = parser.parse_args()
    return args


def build_model(args, device):
    ModelClass = VQ_models[args.vq_model]
    model = ModelClass(
        image_size=args.image_size,
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_l2_norm=args.codebook_l2_norm,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        vq_loss_ratio=args.vq_loss_ratio,
        kl_loss_weight=args.kl_loss_weight,
        dropout_p=args.dropout_p,
        enc_type=args.enc_type,
        encoder_model=args.encoder_model,
        dec_type=args.dec_type,
        decoder_model=args.decoder_model,
        num_latent_tokens=args.num_latent_tokens,
        enc_tuning_method=args.encoder_tuning_method,
        dec_tuning_method=args.decoder_tuning_method,
        enc_pretrained=args.encoder_pretrained,
        dec_pretrained=args.decoder_pretrained,
        enc_patch_size=args.encoder_patch_size,
        dec_patch_size=args.decoder_patch_size,
        tau=args.tau,
        repa=args.repa,
        repa_model=args.repa_model,
        repa_patch_size=args.repa_patch_size,
        repa_proj_dim=args.repa_proj_dim,
        repa_loss_weight=args.repa_loss_weight,
        repa_align=args.repa_align,
        num_codebooks=args.num_codebooks,
        enc_token_drop=args.enc_token_drop,
        enc_token_drop_max=args.enc_token_drop_max,
        cls_recon=args.cls_recon,
        cls_recon_weight=args.cls_recon_weight,
        aux_dec_model=args.aux_decoder_model,
        aux_loss_mask=args.aux_loss_mask,
        aux_hog_dec=args.aux_hog_decoder,
        aux_dino_dec=args.aux_dino_decoder,
        aux_clip_dec=args.aux_clip_decoder,
        aux_supcls_dec=args.aux_supcls_decoder,
        to_pixel=args.to_pixel,
    ).to(device)
    model.eval()

    weights = torch.load(args.weights, map_location="cpu")
    state_dict = weights.get("ema", weights.get("model", weights))
    state_dict = load_model_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    return model


def load_image(path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False),
    ])
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return tensor


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.global_seed)

    model = build_model(args, device)
    images = load_image(args.image, args.image_size).to(device)

    with torch.no_grad():
        quant, _, info = model.encode(images)
        cls_target = getattr(model, "repa_cls_target", None)
        recon = model.decode(quant, x=images, h=images.shape[2], w=images.shape[3]).cpu()
        cls_pred = getattr(model, "dec_cls_pred", None)

    ids = info[2].squeeze(1).contiguous()
    print(ids)
    recon_from_ids = model.decode_from_ids(ids).cpu()

    recon_image = recon[0]
    recon_image = torch.clamp((recon_image + 1) / 2, 0, 1)
    recon_np = (recon_image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(recon_np).save(args.output)

    if cls_target is not None and cls_pred is not None and model.cls_decoder_projector is not None:
        pred_proj = model.cls_decoder_projector(cls_pred).squeeze(0)
        tgt_proj = cls_target.to(pred_proj.device, dtype=pred_proj.dtype).squeeze(0)
        print(pred_proj)
        print(tgt_proj)
        pred_norm = torch.nn.functional.normalize(pred_proj, dim=-1)
        tgt_norm = torch.nn.functional.normalize(tgt_proj, dim=-1)
        mse = torch.mean((pred_proj - tgt_proj) ** 2).item()
        print(pred_norm)
        print(tgt_norm)
        cosine = torch.nn.functional.cosine_similarity(pred_norm, tgt_norm, dim=-1).mean().item()
        print(f"CLS L2: {mse:.6f}, CLS cosine: {cosine:.6f}")
    else:
        print("CLS targets/predictions not available; skipping CLS metrics.")
    
    diff = torch.mean(torch.abs(recon - recon_from_ids)).item()
    print(f"Saved reconstruction to {args.output}")
    print(f"Mean absolute difference between decode() and decode_from_ids(): {diff:.6e}")


if __name__ == "__main__":
    main()
