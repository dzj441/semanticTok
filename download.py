#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download a timm model's pretrained .pth weights to a specified directory.

Usage:
  python download_timm_pth.py --model vit_large_patch14_dinov2.lvd142m --outdir ./weights
  # 可自定义文件名：
  python download_timm_pth.py --model vit_large_patch14_dinov2.lvd142m --outdir ./weights --filename vit_large_patch14_dinov2_lvd142m.pth
"""

import argparse
import os
import sys
import shutil
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="timm model name, e.g. vit_large_patch14_dinov2.lvd142m")
    parser.add_argument("--outdir", required=True, help="destination directory to store the .pth")
    parser.add_argument("--filename", default=None, help="optional output filename; default uses source basename")
    parser.add_argument("--force", action="store_true", help="overwrite if the file already exists")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    try:
        import timm
    except Exception as e:
        print("ERROR: timm is not installed. `pip install timm` on the online machine.", file=sys.stderr)
        sys.exit(1)

    # Try to resolve a direct URL from timm's pretrained config
    url = None
    repo_id = None
    hf_fname = None
    ckpt_path = None

    try:
        m = timm.create_model(args.model, pretrained=False)
    except Exception as e:
        print(f"ERROR: cannot instantiate model '{args.model}': {e}", file=sys.stderr)
        sys.exit(1)

    # timm >=0.9 uses `pretrained_cfg`; older uses `default_cfg`
    cfg = getattr(m, "pretrained_cfg", None)
    if not cfg:
        cfg = getattr(m, "default_cfg", {})  # fallback for older timm

    # cfg can be a dict-like, try common keys
    if isinstance(cfg, dict):
        url = cfg.get("url") or cfg.get("checkpoint_url") or cfg.get("weights_url")
        # HF fallback keys (not always present)
        repo_id = cfg.get("hf_hub_id") or cfg.get("hf_hub_id_timm")
        hf_fname = cfg.get("hf_hub_filename") or cfg.get("filename")
    else:
        # very defensive fallback
        try:
            url = cfg.get("url", None)
        except Exception:
            url = None

    # 1) Preferred: download via exact URL into cache, then copy out
    if url:
        try:
            from timm.models._hub import download_cached_file
            print(f"[info] downloading from URL (timm cfg): {url}")
            ckpt_path = download_cached_file(url, check_hash=True, progress=True)
        except Exception as e:
            print(f"[warn] direct URL download failed: {e}", file=sys.stderr)
            ckpt_path = None

    # 2) HuggingFace fallback (if provided by timm cfg)
    if ckpt_path is None and repo_id:
        try:
            from huggingface_hub import hf_hub_download
            if not hf_fname:
                # a common default name used by many repos
                hf_fname = "pytorch_model.bin"
            print(f"[info] downloading from HF: repo_id={repo_id}, filename={hf_fname}")
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=hf_fname,
                local_dir=os.path.expanduser("~/.cache/timm_hf"),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            print(f"[warn] HF download failed: {e}", file=sys.stderr)
            ckpt_path = None

    # 3) Last resort: trigger timm's own pretrained download to ~/.cache/torch/hub/checkpoints
    if ckpt_path is None:
        try:
            print("[info] triggering timm to download via create_model(pretrained=True)...")
            _ = timm.create_model(args.model, pretrained=True)
            cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
            # pick the most recent .pth in cache (works best in a clean env)
            candidates = sorted(glob.glob(os.path.join(cache_dir, "*.pth")), key=os.path.getmtime)
            if not candidates:
                raise RuntimeError(f"No .pth found in {cache_dir} after download")
            ckpt_path = candidates[-1]
            print(f"[info] picked cached checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"ERROR: failed to obtain checkpoint via timm: {e}", file=sys.stderr)
            sys.exit(1)

    # Decide destination filename
    dst_name = args.filename if args.filename else os.path.basename(ckpt_path)
    dst_path = os.path.join(args.outdir, dst_name)

    if os.path.exists(dst_path) and not args.force:
        print(f"[info] destination exists, use --force to overwrite: {dst_path}")
    else:
        shutil.copy2(ckpt_path, dst_path)
        print(f"[ok] saved to: {dst_path}")

    # Optional: sanity check that torch can read it (will fail on non-.pth)
    try:
        import torch
        _ = torch.load(dst_path, map_location="cpu")
        print("[ok] torch.load check passed.")
    except Exception as e:
        print(f"[warn] torch.load failed (file may not be a pure .pth or has different format): {e}", file=sys.stderr)

if __name__ == "__main__":
    main()