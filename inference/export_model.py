import argparse
import os

import torch

from utils.misc import load_model_state_dict, str2bool


def parse_args():
    parser = argparse.ArgumentParser(description="Extract model weights from training checkpoint")
    parser.add_argument("--checkpoint", type=str, default="experiments/tokenizer/exp002-simvq-bl-128/checkpoints/0250000.pt", help="Path to training checkpoint (.pt)")
    parser.add_argument("--out", type=str,default= "experiments/tokenizer/exp002-simvq-bl-128/checkpoints/0250000.pth", help="Destination .pth path for model-only weights")
    parser.add_argument("--prefer-ema", type=str2bool, default=True, help="Prefer EMA weights if available")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    state = None
    if args.prefer_ema and isinstance(ckpt, dict) and "ema" in ckpt:
        state = ckpt["ema"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    state = load_model_state_dict(state)
    payload = {"model": state}
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(payload, args.out)
    print(f"Saved model weights to {args.out}")


if __name__ == "__main__":
    main()
