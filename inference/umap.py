import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
import random

sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import umap
except ImportError as exc:
    raise ImportError(
        "UMAP visualization requires the 'umap-learn' package. "
        "Install it via `pip install umap-learn`."
    ) from exc

from modelling.tokenizer import VQ_models
from train.train_tokenizer import build_parser
from utils.data import center_crop_arr
from utils.misc import load_model_state_dict

SOURCE_ALIASES = {
    "siglip": "siglip",
    "teacher": "siglip",
    "latent": "siglip",
    "recon": "recon",
    "decoded": "recon",
    "decode": "recon",
    "quantized": "latent12",
    "latent12": "latent12",
    "quant": "latent12",
}
DEFAULT_SOURCES = ["siglip", "recon", "latent12"]
SIGLIP_SOURCES = {"siglip", "recon"}
LATENT_SOURCES = {"latent12"}


def parse_args():
    parser = build_parser()
    parser.add_argument("--weights", type=str, default="weights/model.pth", help="Model checkpoint to load.")
    parser.add_argument("--device", type=str, default=None, help="Device spec such as cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap the number of images processed.")
    parser.add_argument(
        "--feature-sources",
        type=str,
        default="siglip,recon,latent12",
        help="Comma separated sources: siglip (teacher), recon (decoder output), latent12 (quantized codes).",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="token",
        choices=["token", "mean"],
        help="Whether to treat every token as a UMAP sample or pool per-image.",
    )
    parser.add_argument(
        "--max-features-per-source",
        type=int,
        default=None,
        help="Optional limit per source after pooling (useful to subsample large token sets).",
    )
    parser.add_argument("--umap-components", type=int, default=2, help="Target embedding dimensionality.")
    parser.add_argument("--umap-neighbors", type=int, default=25, help="UMAP n_neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--umap-metric", type=str, default="cosine", help="UMAP metric.")
    parser.add_argument("--umap-seed", type=int, default=42, help="Random seed for UMAP.")
    parser.add_argument("--save-embeddings", type=str, default="results/umap_embeddings.npz", help="Where to store raw features/metadata.")
    parser.add_argument("--plot-path", type=str, default="results/umap_plot.png", help="Where to save the UMAP scatter plot.")
    parser.add_argument(
        "--color-by",
        type=str,
        default="source",
        choices=["source", "class"],
        help="Color the scatter plot by feature source or dataset class.",
    )
    parser.add_argument("--marker-size", type=float, default=6.0, help="Matplotlib marker size.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip plotting, only dump embeddings/metadata.")
    parser.add_argument("--skip-umap", action="store_true", help="Skip running UMAP (only saves raw features).")
    parser.add_argument("--num-classes", type=int, default=20, help="Randomly sample this many classes (<=0 disables sampling).")
    parser.add_argument("--samples-per-class", type=int, default=25, help="Randomly sample this many images per selected class.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Random seed for class/image sampling (defaults to --umap-seed).")
    parser.add_argument("--global-pooling", type=str, default="siglip", choices=["siglip", "mean"], help="Pooling strategy for 1024-d features.")
    parser.add_argument("--latent-pooling", type=str, default="mean", choices=["mean"], help="Pooling for 12-d latent tokens.")
    parser.add_argument("--pca-dim", type=int, default=64, help="Apply PCA to this dimension before UMAP when > 0 and feature dim exceeds it.")
    parser.add_argument("--skip-pca", action="store_true", help="Disable PCA preprocessing before UMAP.")

    args = parser.parse_args()
    if args.config and os.path.isfile(args.config):
        from ruamel import yaml

        with open(args.config, "r", encoding="utf-8") as f:
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
        aux_dec_model=args.aux_decoder_model,
        aux_loss_mask=args.aux_loss_mask,
        aux_hog_dec=args.aux_hog_decoder,
        aux_dino_dec=args.aux_dino_decoder,
        aux_clip_dec=args.aux_clip_decoder,
        aux_supcls_dec=args.aux_supcls_decoder,
        to_pixel=args.to_pixel,
    ).to(device)
    model.eval()
    return model


def load_model_weights(model, weights_path):
    payload = torch.load(weights_path, map_location="cpu")
    state = payload.get("model", payload)
    state = load_model_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[UMAP] Missing keys: {missing}")
    if unexpected:
        print(f"[UMAP] Unexpected keys: {unexpected}")


def build_dataloader(args):
    eval_root = args.eval_data_path if args.eval_data_path else args.data_path
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    dataset = ImageFolder(eval_root, transform=transform)
    subset_indices, subset_report = select_subset_indices(
        dataset,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        seed=args.sample_seed if args.sample_seed is not None else args.umap_seed,
    )
    if subset_indices is not None:
        subset = Subset(dataset, subset_indices)
    else:
        subset = dataset
    loader = DataLoader(
        subset,
        batch_size=args.batch_size or args.global_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, dataset, subset_report


def select_subset_indices(dataset, num_classes, samples_per_class, seed=None):
    if num_classes is None or num_classes <= 0 or samples_per_class is None or samples_per_class <= 0:
        return None, None
    rng = random.Random(seed)
    class_to_indices = defaultdict(list)
    for idx, (_, cls) in enumerate(dataset.samples):
        class_to_indices[cls].append(idx)
    available = [cls for cls, idxs in class_to_indices.items() if idxs]
    if not available:
        return None, None
    select_k = min(num_classes, len(available))
    chosen_classes = rng.sample(available, select_k)
    subset_indices = []
    report = []
    for cls in chosen_classes:
        indices = class_to_indices[cls]
        take = min(samples_per_class, len(indices))
        if len(indices) <= take:
            chosen = indices.copy()
            rng.shuffle(chosen)
        else:
            chosen = rng.sample(indices, take)
        subset_indices.extend(chosen)
        report.append((cls, dataset.classes[cls], len(chosen)))
    rng.shuffle(subset_indices)
    report.sort(key=lambda x: x[0])
    return subset_indices, report


def _normalize_tensor(tensor):
    if tensor is None:
        return None
    if tensor.dim() == 4:
        tensor = tensor.flatten(2).transpose(1, 2)
    return tensor


def normalize_sources(spec: str):
    raw = [s.strip().lower() for s in spec.split(",")] if spec else []
    raw = [s for s in raw if s]
    if not raw:
        raw = DEFAULT_SOURCES
    normalized = set()
    for name in raw:
        normalized.add(SOURCE_ALIASES.get(name, name))
    return normalized


def pool_tensor_for_source(tensor, source_name, args, siglip_pooler=None):
    if tensor is None:
        return None
    tensor = _normalize_tensor(tensor)
    if tensor is None:
        return None

    if source_name in LATENT_SOURCES:
        return tensor.mean(dim=1)

    if source_name in SIGLIP_SOURCES and args.global_pooling == "siglip" and siglip_pooler is not None:
        try:
            return siglip_pooler(tensor)
        except Exception as exc:
            print(f"[UMAP] SigLIP pooling failed for source '{source_name}': {exc}. Falling back to mean pooling.")

    if source_name in SIGLIP_SOURCES and args.global_pooling == "mean":
        return tensor.mean(dim=1)

    if args.pooling == "mean":
        return tensor.mean(dim=1)
    return tensor.reshape(-1, tensor.size(-1))


def collect_embeddings(model, loader, device, args, siglip_pooler=None):
    wanted = normalize_sources(args.feature_sources)
    per_source_cap = args.max_features_per_source

    features = []
    sources = []
    image_indices = []
    class_labels = []
    token_indices = []
    per_source_counts = Counter()

    processed = 0
    global_index = 0

    need_recon = "recon" in wanted
    need_siglip = "siglip" in wanted
    need_latent = "latent12" in wanted

    progress = tqdm(loader, desc="Collecting", leave=False)
    for images, labels in progress:
        batch_size = images.size(0)
        remaining = None
        if args.max_samples is not None:
            remaining = max(args.max_samples - processed, 0)
            if remaining == 0:
                break
            if batch_size > remaining:
                images = images[:remaining]
                labels = labels[:remaining]
                batch_size = remaining

        images = images.to(device, non_blocking=True)
        labels_np = labels.cpu().numpy()
        with torch.no_grad():
            (quant_tokens, latent_tokens), _, _ = model.encode(images)
            recon_tokens = None
            if need_recon:
                recon_tokens, _ = model.decode(quant_tokens)

        sample_ids = np.arange(global_index, global_index + batch_size, dtype=np.int64)
        global_index += batch_size
        processed += batch_size

        source_tensors = {
            "siglip": latent_tokens if need_siglip else None,
            "recon": recon_tokens if need_recon else None,
            "latent12": quant_tokens if need_latent else None,
        }

        for source_name, tensor in source_tensors.items():
            if source_name not in wanted or tensor is None:
                continue
            pooled = pool_tensor_for_source(tensor, source_name, args, siglip_pooler=siglip_pooler)
            if pooled is None:
                continue

            if source_name in SIGLIP_SOURCES or source_name in LATENT_SOURCES or args.pooling == "mean":
                img_ids = sample_ids
                cls_ids = labels_np
                tok_ids = np.full(pooled.size(0), -1, dtype=np.int64)
            else:
                # token-level entries
                b, tokens = tensor.size(0), tensor.size(1)
                img_ids = np.repeat(sample_ids, tokens)
                cls_ids = np.repeat(labels_np, tokens)
                tok_ids = np.tile(np.arange(tokens, dtype=np.int64), b)

            pooled_np = pooled.detach().float().cpu().numpy().astype(np.float32, copy=False)
            total = pooled_np.shape[0]
            if per_source_cap is not None:
                remaining_quota = per_source_cap - per_source_counts[source_name]
                if remaining_quota <= 0:
                    continue
                if total > remaining_quota:
                    pooled_np = pooled_np[:remaining_quota]
                    img_ids = img_ids[:remaining_quota]
                    cls_ids = cls_ids[:remaining_quota]
                    tok_ids = tok_ids[:remaining_quota]
                    total = remaining_quota

            features.append(pooled_np)
            sources.extend([source_name] * total)
            image_indices.extend(img_ids.tolist())
            class_labels.extend(cls_ids.tolist())
            token_indices.extend(tok_ids.tolist())
            per_source_counts[source_name] += total

        if args.max_samples is not None and processed >= args.max_samples:
            break

    if not features:
        raise RuntimeError("No features collected. Check --feature-sources or dataset contents.")

    feature_matrix = np.concatenate(features, axis=0)
    metadata = {
        "sources": np.array(sources, dtype=object),
        "image_indices": np.array(image_indices, dtype=np.int64),
        "class_labels": np.array(class_labels, dtype=np.int64),
        "token_indices": np.array(token_indices, dtype=np.int64),
    }
    return feature_matrix, metadata, per_source_counts


def run_umap(feature_matrix, args):
    reducer = umap.UMAP(
        n_components=args.umap_components,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.umap_seed,
    )
    embedding = reducer.fit_transform(feature_matrix)
    return embedding


def apply_pca(features: np.ndarray, target_dim: int):
    if target_dim is None or target_dim <= 0:
        return features, None
    feat_dim = features.shape[1]
    if feat_dim <= target_dim:
        return features, None
    centered = features - features.mean(axis=0, keepdims=True)
    # economy SVD
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:target_dim]
    reduced = centered @ components.T
    info = {
        "components": components.astype(np.float32, copy=False),
        "mean": features.mean(axis=0).astype(np.float32, copy=False),
    }
    return reduced.astype(np.float32, copy=False), info


def plot_embedding(embedding, metadata, args):
    if args.umap_components not in (2, 3):
        print(f"[UMAP] Skipping plot for {args.umap_components} components (only 2D/3D supported).")
        return

    color_field = metadata["sources"] if args.color_by == "source" else metadata["class_labels"]
    categories = sorted(set(color_field))
    palette = plt.cm.get_cmap("tab20", max(len(categories), 1))
    color_map = {cat: palette(i % palette.N) for i, cat in enumerate(categories)}

    if args.umap_components == 2:
        fig, ax = plt.subplots(figsize=(8, 8))
        for cat in categories:
            mask = color_field == cat
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=args.marker_size,
                alpha=0.7,
                label=str(cat),
                color=color_map[cat],
            )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title("UMAP projection")
        ax.legend(loc="best", fontsize="small", markerscale=1.5)
        fig.tight_layout()
        Path(args.plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot_path, dpi=300)
        plt.close(fig)
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for cat in categories:
            mask = color_field == cat
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                s=args.marker_size,
                alpha=0.7,
                label=str(cat),
                color=color_map[cat],
            )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_zlabel("UMAP-3")
        ax.set_title("UMAP projection (3D)")
        ax.legend(loc="best", fontsize="small", markerscale=1.2)
        fig.tight_layout()
        Path(args.plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot_path, dpi=300)
        plt.close(fig)


def main():
    args = parse_args()
    if getattr(args, "distributed", False):
        raise NotImplementedError("inference/umap.py currently supports single-process execution only.")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.global_seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.global_seed)

    model = build_model(args, device)
    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    load_model_weights(model, str(weights_path))

    loader, dataset, subset_report = build_dataloader(args)
    if subset_report:
        print("[UMAP] Sampling classes (idx, name, count):")
        for cls_idx, cls_name, count in subset_report:
            print(f"  - {cls_idx:4d} {cls_name}: {count}")

    siglip_pooler = None
    encode_transformer = getattr(model, "encode_transformer", None)
    if encode_transformer is not None and hasattr(encode_transformer, "pool_tokens"):
        siglip_pooler = encode_transformer.pool_tokens
    elif args.global_pooling == "siglip":
        print("[UMAP] Warning: SigLIP pooling requested but pool_tokens not available. Falling back to mean pooling.")

    features, metadata, counters = collect_embeddings(model, loader, device, args, siglip_pooler=siglip_pooler)
    pca_info = None
    if (not args.skip_pca) and args.pca_dim and features.shape[1] > args.pca_dim:
        features, pca_info = apply_pca(features, args.pca_dim)
        print(f"[UMAP] Applied PCA to {features.shape[1]} dims (target {args.pca_dim}).")

    npz_path = Path(args.save_embeddings)
    Path(args.save_embeddings).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.save_embeddings,
        features=features,
        sources=metadata["sources"],
        image_indices=metadata["image_indices"],
        class_labels=metadata["class_labels"],
        token_indices=metadata["token_indices"],
        class_to_idx=dataset.class_to_idx,
        pca_mean=None if pca_info is None else pca_info["mean"],
        pca_components=None if pca_info is None else pca_info["components"],
    )
    print(f"[UMAP] Saved feature matrix {features.shape} to {args.save_embeddings}")
    for source_name, count in counters.items():
        print(f"[UMAP] {source_name}: {count} samples")

    embedding = None
    if not args.skip_umap:
        embedding = run_umap(features, args)
        np.savez(
            npz_path.with_name(npz_path.stem + "_umap.npz"),
            embedding=embedding,
            sources=metadata["sources"],
            image_indices=metadata["image_indices"],
            class_labels=metadata["class_labels"],
            token_indices=metadata["token_indices"],
        )
        print(f"[UMAP] Saved UMAP embedding to {npz_path.with_name(npz_path.stem + '_umap.npz')}")

    if embedding is not None and not args.skip_plot:
        plot_embedding(embedding, metadata, args)
        print(f"[UMAP] Plot stored at {args.plot_path}")


if __name__ == "__main__":
    main()
