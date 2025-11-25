import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.append(os.getcwd())
from modelling.tokenizer import VQ_models
from train.train_tokenizer import build_parser
from utils.data import center_crop_arr
from utils.misc import load_model_state_dict


# ========================
# 全局配置（按需修改）
# ========================
CONFIG_PATH = "configs/simvq-bl-128-eval.yaml"
WEIGHTS_PATH = "weights/model.pth"
DEVICE = None  # None -> 自动选择
DISTRIBUTED = False
PER_PROC_BATCH_SIZE = None
BATCH_SIZE = 32
GLOBAL_BATCH_SIZE = None
GLOBAL_SEED = None
NUM_WORKERS = None
SAVE_PNG = False
SAMPLE_DIR = "semantic_eval_samples"
NPZ_COUNT = 50000
COMPUTE_FID = True
FID_WEIGHTS_PATH = "weights/weights-inception-2015-12-05-6726825d.pth"
DATA_PATH_OVERRIDE = None
EVAL_DATA_PATH_OVERRIDE = None


def build_npz_from_folder(folder: str, limit: int = 50000):
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Sample directory '{folder}' not found.")

    png_files = sorted(folder_path.glob("*.png"))
    if not png_files:
        raise RuntimeError(f"No PNG files found in {folder} to build NPZ.")

    if limit:
        png_files = png_files[:limit]

    samples = []
    for png in tqdm(png_files, desc="Building NPZ", leave=False):
        samples.append(np.asarray(Image.open(png)).astype(np.uint8))
    arr = np.stack(samples)
    out_path = folder_path.with_suffix(".npz")
    np.savez(out_path, arr_0=arr)
    print(f"[Eval] Saved NPZ with {arr.shape[0]} samples to {out_path}")


def load_eval_args():
    parser = build_parser()

    if CONFIG_PATH:
        if not os.path.isfile(CONFIG_PATH):
            raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
        from ruamel import yaml

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**(config_args or {}))

    args = parser.parse_args([])

    args.config = CONFIG_PATH
    args.weights = WEIGHTS_PATH
    args.device = DEVICE
    args.distributed = DISTRIBUTED
    args.per_proc_batch_size = PER_PROC_BATCH_SIZE
    if BATCH_SIZE is not None:
        args.batch_size = BATCH_SIZE
    if GLOBAL_BATCH_SIZE is not None:
        args.global_batch_size = GLOBAL_BATCH_SIZE
    if GLOBAL_SEED is not None:
        args.global_seed = GLOBAL_SEED
    if NUM_WORKERS is not None:
        args.num_workers = NUM_WORKERS
    args.save_png = SAVE_PNG
    args.sample_dir = SAMPLE_DIR
    args.npz_count = NPZ_COUNT
    args.fid = COMPUTE_FID
    args.fid_weights = FID_WEIGHTS_PATH
    if DATA_PATH_OVERRIDE is not None:
        args.data_path = DATA_PATH_OVERRIDE
    if EVAL_DATA_PATH_OVERRIDE is not None:
        args.eval_data_path = EVAL_DATA_PATH_OVERRIDE
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
        enc_local_ckpt=args.encoder_local_ckpt,
        dec_local_ckpt=args.decoder_local_ckpt,
        repa_local_ckpt=args.repa_local_ckpt,
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
        semantic_target_layers=args.semantic_target_layers,
        semantic_qam_layers=args.semantic_qam_layers,
        aux_dec_model=args.aux_decoder_model,
        aux_loss_mask=args.aux_loss_mask,
        aux_hog_dec=args.aux_hog_decoder,
        aux_dino_dec=args.aux_dino_decoder,
        aux_clip_dec=args.aux_clip_decoder,
        aux_supcls_dec=args.aux_supcls_decoder,
        to_pixel=args.to_pixel,
        transformer_config=args.transformer_config,
        codebook_slots_embed_dim=args.codebook_slots_embed_dim,
    ).to(device)
    model.eval()
    return model


def load_model_weights(model, weights_path):
    payload = torch.load(weights_path, map_location="cpu")
    state = payload.get("model", payload)
    state = load_model_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Eval] Missing keys: {missing}")
    if unexpected:
        print(f"[Eval] Unexpected keys: {unexpected}")


def build_dataloader(args, distributed=False, world_size=1, rank=0):
    eval_root = args.eval_data_path if args.eval_data_path else args.data_path
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    dataset = ImageFolder(eval_root, transform=transform)
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if args.per_proc_batch_size is not None:
            batch_size = args.per_proc_batch_size
        elif args.batch_size is not None:
            batch_size = max(1, args.batch_size // world_size)
        else:
            batch_size = max(1, args.global_batch_size // world_size)
    else:
        sampler = None
        batch_size = args.batch_size or args.global_batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, len(dataset), sampler


def evaluate(model, loader, device, *, distributed=False, rank=0, world_size=1, fid_metric=None, save_png=False, sample_dir=None):
    mse_sum = 0.0
    l1_sum = 0.0
    total_count = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    psnr_count = 0

    next_png_index = rank

    progress = tqdm(loader, desc="Eval", leave=False) if (not distributed or rank == 0) else loader
    for images, _ in progress:
        images = images.to(device)
        with torch.no_grad():
            recon, _, _ = model(images)

        batch_size = images.size(0)
        mse = F.mse_loss(recon, images, reduction="mean").item()
        l1 = F.l1_loss(recon, images, reduction="mean").item()
        mse_sum += mse * batch_size
        l1_sum += l1 * batch_size
        total_count += batch_size

        recon_np = torch.clamp((recon + 1) / 2, 0, 1).permute(0, 2, 3, 1).cpu().numpy()
        gt_np = torch.clamp((images + 1) / 2, 0, 1).permute(0, 2, 3, 1).cpu().numpy()

        for rec, gt in zip(recon_np, gt_np):
            psnr_sum += psnr_loss(gt, rec, data_range=1.0)
            ssim_sum += ssim_loss(gt, rec, channel_axis=-1, data_range=1.0)
            psnr_count += 1

        if fid_metric is not None:
            real_uint8 = torch.clamp(images * 127.5 + 128, 0, 255).round().to(torch.uint8)
            fake_uint8 = torch.clamp(recon * 127.5 + 128, 0, 255).round().to(torch.uint8)
            fid_metric.update(real_uint8, real=True)
            fid_metric.update(fake_uint8, real=False)

        if save_png and sample_dir is not None:
            recon_uint8 = (np.clip(recon_np, 0, 1) * 255.0).round().astype(np.uint8)
            for img_np in recon_uint8:
                filename = Path(sample_dir) / f"{next_png_index:06d}.png"
                Image.fromarray(img_np).save(filename)
                next_png_index += world_size

    if distributed:
        stats = torch.tensor(
            [mse_sum, l1_sum, psnr_sum, ssim_sum, float(total_count), float(psnr_count)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats)
        mse_sum, l1_sum, psnr_sum, ssim_sum, total_count, psnr_count = stats.tolist()
    metrics = {
        "mse": float(mse_sum / total_count) if total_count else 0.0,
        "l1": float(l1_sum / total_count) if total_count else 0.0,
        "psnr": float(psnr_sum / psnr_count) if psnr_count else 0.0,
        "ssim": float(ssim_sum / psnr_count) if psnr_count else 0.0,
    }

    if fid_metric is not None:
        fid_score = fid_metric.compute().detach().item()
        metrics["fid"] = fid_score

    return metrics


def main():
    args = load_eval_args()

    distributed = args.distributed
    rank = 0
    world_size = 1
    local_rank = 0

    if distributed:
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())) if torch.cuda.is_available() else 0
    else:
        local_rank = 0

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

    torch.manual_seed(args.global_seed + rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.global_seed + rank)

    model = build_model(args, device)
    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    load_model_weights(model, str(weights_path))

    loader, dataset_len, sampler = build_dataloader(args, distributed=distributed, world_size=world_size, rank=rank)
    if sampler is not None:
        sampler.set_epoch(0)

    fid_metric = None
    if args.fid:
        fid_metric = FrechetInceptionDistance(normalize=False, feature_extractor_weights_path=args.fid_weights)
        fid_metric = fid_metric.to(device)
        fid_metric.reset()

    if args.save_png:
        if (not distributed) or (rank == 0):
            os.makedirs(args.sample_dir, exist_ok=True)
        if distributed:
            dist.barrier()

    metrics = evaluate(
        model,
        loader,
        device,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        save_png=args.save_png,
        sample_dir=args.sample_dir,
        fid_metric=fid_metric,
    )

    if distributed:
        dist.barrier()

    if (not distributed) or (rank == 0):
        print("========== Semantic Evaluation Summary ==========")
        print(f"Dataset size: {dataset_len}")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")

        if args.save_png:
            build_npz_from_folder(args.sample_dir, limit=args.npz_count)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
