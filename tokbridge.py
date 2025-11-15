# reconstruction_softvq_tbquant.py
import warnings
warnings.filterwarnings('ignore', message='Overwriting.*in registry')

import os, json, math, multiprocessing as mp
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
from tqdm import tqdm

# === SoftVQ 模型 ===
from modelling.tokenizer import SoftVQModel  # 你提供的实现

# === 指标：rFID & IS ===
from torchmetrics.image.fid import FrechetInceptionDistance
import torch_fidelity
PATH = "../weights/pt_inception-2015-12-05-6726825d.pth"
# ---------- 基础工具 ----------
class CenterCropTransform:
    def __init__(self, image_size: int):
        self.image_size = image_size
    def __call__(self, pil_image):
        while min(*pil_image.size) >= 2 * self.image_size:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
        scale = self.image_size / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - self.image_size) // 2
        crop_x = (arr.shape[1] - self.image_size) // 2
        return Image.fromarray(arr[crop_y: crop_y + self.image_size, crop_x: crop_x + self.image_size])

class ClasswiseImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.samples_by_class = {}
        for class_idx in range(len(self.classes)):
            cls_idx = [i for i, (_, t) in enumerate(self.samples) if t == class_idx]
            self.samples_by_class[class_idx] = cls_idx

def save_image(t: torch.Tensor, path: str):
    img = (t + 1) / 2.0
    img = torch.clamp(img, 0, 1)
    img = img.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

# ---------- TokenBridge 式后量化（保持原逻辑） ----------
from scipy.stats import norm
def post_training_quantize(x: torch.Tensor, bits: int, min_range: float, max_range: float, std_range: float = 3.0):
    """
    高斯等概率 + 截断 + 区间期望反量化（与 TokenBridge 相同）
    x: 任意形状的连续张量（这里会是 SoftVQ 的 latent，(B, L, D) 或 (B, C, H', W')）
    """
    if bits < 0:
        return x, x
    n = 2 ** bits
    device, dtype = x.device, x.dtype

    probs = torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    boundaries = torch.tensor(norm.ppf(probs.cpu()), device=device, dtype=dtype).clamp(-std_range, std_range)

    def truncated_normal_mean(a, b):
        sqrt_2 = math.sqrt(2)
        sqrt_2pi = math.sqrt(2 * math.pi)
        phi_a = torch.exp(-0.5 * a**2) / sqrt_2pi
        phi_b = torch.exp(-0.5 * b**2) / sqrt_2pi
        Phi_a = 0.5 * (1 + torch.erf(a / sqrt_2))
        Phi_b = 0.5 * (1 + torch.erf(b / sqrt_2))
        denom = torch.where((Phi_b - Phi_a) == 0,
                            torch.tensor(1e-10, device=device, dtype=dtype),
                            (Phi_b - Phi_a))
        return (phi_a - phi_b) / denom

    recons = []
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1]
        recons.append(truncated_normal_mean(a, b))
    reconstruction_values = torch.stack(recons, dim=0).to(device=device, dtype=dtype)  # (n,)

    # 线性映射到 [-std_range, std_range]
    x_norm = (x - min_range) / (max_range - min_range) * (2 * std_range) - std_range
    x_clamped = x_norm.clamp(-std_range, std_range)

    # 最近重构值
    d = (x_clamped.unsqueeze(-1) - reconstruction_values).abs()
    idx = d.argmin(dim=-1)

    values = (reconstruction_values + std_range) / (2 * std_range) * (max_range - min_range) + min_range
    dequant = values[idx]
    return idx, dequant

# ---------- 指标（抽成函数，SoftVQ 风格） ----------
class RfidMeter:
    """
    SoftVQ 脚本风格的 rFID：用 torchmetrics 的 FID，
    逐批 update(real=True/False)，最后 compute()。
    传入的张量应为 uint8、范围 [0,255]、形状 (B,3,H,W)。
    """
    def __init__(self, device="cuda"):
        self.metric = FrechetInceptionDistance(normalize=False,feature_extractor_weights_path=PATH).to(device)
        self.device = device
    @torch.no_grad()
    def update(self, real_uint8: torch.Tensor, fake_uint8: torch.Tensor):
        self.metric.update(real_uint8.to(self.device), real=True)
        self.metric.update(fake_uint8.to(self.device), real=False)
    @torch.no_grad()
    def compute(self) -> float:
        return float(self.metric.compute().detach().cpu())

class LatentStats:
    def __init__(self, hist_bins=201):
        self.hist_bins = hist_bins
        self.hist_min = None
        self.hist_max = None
        self.hist_counts = None  # torch.IntTensor[bins]
        self.total = 0
        self.xmin = None
        self.xmax = None
        self.index_hist = None   # torch.IntTensor[n_levels]
        self.n_levels = None

    @torch.no_grad()
    def update_values(self, x: torch.Tensor, hist_min: float, hist_max: float):
        x = x.detach().reshape(-1).to('cpu')
        if x.numel() == 0: return
        self.total += x.numel()
        self.xmin = x.min() if self.xmin is None else torch.minimum(self.xmin, x.min())
        self.xmax = x.max() if self.xmax is None else torch.maximum(self.xmax, x.max())
        if self.hist_counts is None:
            self.hist_min, self.hist_max = hist_min, hist_max
            self.hist_counts = torch.histc(x, bins=self.hist_bins, min=hist_min, max=hist_max).to(torch.int64)
        else:
            self.hist_counts += torch.histc(x, bins=self.hist_bins, min=self.hist_min, max=self.hist_max).to(torch.int64)

    @torch.no_grad()
    def update_indices(self, q_idx: torch.Tensor, n_levels: int):
        q_idx = q_idx.detach().reshape(-1).to('cpu')
        if self.index_hist is None:
            self.n_levels = n_levels
            self.index_hist = torch.zeros(n_levels, dtype=torch.int64)
        self.index_hist += torch.bincount(q_idx, minlength=n_levels)

    def _quantile_from_hist(self, q: float) -> float:
        cdf = self.hist_counts.cumsum(0).to(torch.float64)
        total = cdf[-1].clone()
        if total <= 0:
            return float('nan')
        cdf = cdf / total
        idx = torch.searchsorted(cdf, torch.tensor(q, dtype=cdf.dtype))
        idx = torch.clamp(idx, 0, self.hist_bins - 1)
        bin_w = (self.hist_max - self.hist_min) / self.hist_bins
        return float(self.hist_min + (idx + 0.5) * bin_w)


    def finalize(self) -> dict:
        if self.hist_counts is None or self.total == 0:
            return {}
        bins = torch.arange(self.hist_bins, dtype=torch.float64)
        bin_w = (self.hist_max - self.hist_min) / self.hist_bins
        mids = (self.hist_min + (bins + 0.5) * bin_w).to(torch.float64)
        cnt = self.hist_counts.to(torch.float64)
        N = cnt.sum().clamp_min(1.0)

        mean = (mids * cnt).sum() / N
        var  = ((mids - mean) ** 2 * cnt).sum() / (N - 1.0).clamp_min(1.0)
        std  = var.sqrt().clamp_min(1e-12)
        m3   = ((mids - mean) ** 3 * cnt).sum() / N
        m4   = ((mids - mean) ** 4 * cnt).sum() / N
        skew = (m3 / std**3).item()
        kurt = (m4 / std**4).item() - 3.0

        q01  = self._quantile_from_hist(0.01)
        q05  = self._quantile_from_hist(0.05)
        q50  = self._quantile_from_hist(0.50)
        q95  = self._quantile_from_hist(0.95)
        q99  = self._quantile_from_hist(0.99)
        q999 = self._quantile_from_hist(0.999)

        R_99  = max(abs(q01),  abs(q99))
        R_999 = max(abs(self._quantile_from_hist(0.001)), abs(q999))

        used = int((self.index_hist > 0).sum().item()) if self.index_hist is not None else None
        wasted_ratio = 1.0 - used / float(self.n_levels) if used is not None else None

        normality_hint = "approx_gaussian" if (abs(skew) < 0.1 and abs(kurt) < 0.5) else "non_gaussian"

        return {
            "count": int(self.total),
            "min": float(self.xmin), "max": float(self.xmax),
            "mean": float(mean), "std": float(std),
            "skewness": float(skew), "excess_kurtosis": float(kurt),
            "normality_hint": normality_hint,
            "q01": q01, "q05": q05, "q50": q50, "q95": q95, "q99": q99, "q999": q999,
            "recommend": {"symmetric_range@99%": R_99, "symmetric_range@99.9%": R_999},
            "index_bins_used": used, "index_bins_total": self.n_levels,
            "index_bins_wasted_ratio": wasted_ratio
        }


class PerChannelLatentStats:
    def __init__(self, hist_bins=201):
        self.hist_bins = hist_bins
        self.stats = None  # list[LatentStats]
        self.D = None

    def ensure(self, D: int):
        if self.stats is None:
            self.D = D
            self.stats = [LatentStats(hist_bins=self.hist_bins) for _ in range(D)]

    @torch.no_grad()
    def update_values(self, lat: torch.Tensor, hist_min: float, hist_max: float):
        # lat: (B, L, D)  —— ViT 分支
        self.ensure(lat.shape[-1])
        for c in range(self.D):
            self.stats[c].update_values(lat[..., c], hist_min, hist_max)

    @torch.no_grad()
    def update_indices(self, idx: torch.Tensor, n_levels: int):
        # idx: (B, L, D)
        if self.stats is None:
            return
        for c in range(self.D):
            self.stats[c].update_indices(idx[..., c], n_levels)

    def finalize(self) -> list:
        if self.stats is None:
            return []
        out = []
        for c, st in enumerate(self.stats):
            rep = st.finalize()
            rep["channel"] = c
            out.append(rep)
        return out

def compute_inception_score_from_dir(img_dir: str) -> Tuple[float, float]:
    """
    用 torch_fidelity 计算目录下重建图像的 Inception Score（SoftVQ 脚本常见做法）
    """
    m = torch_fidelity.calculate_metrics(
        input1=img_dir, input2=None,
        isc=True, fid=False, cuda=True, verbose=False,feature_extractor_weights_path=PATH
    )
    return float(m["inception_score_mean"]), float(m["inception_score_std"])

# ---------- 主流程 ----------
def process_classes(rank: int, args, start_cls: int, end_cls: int):
    device = f"cuda:{rank % max(1, torch.cuda.device_count())}"

    # 1) 加载 SoftVQ
    vq = SoftVQModel.from_pretrained(args.vq_model)
    vq.eval().to(device)
    
    qmin, qmax = -args.range, args.range
    global_stats = LatentStats(hist_bins=201)
    per_ch_stats = PerChannelLatentStats(hist_bins=201)
    
    # 2) 数据
    transform = transforms.Compose([
        CenterCropTransform(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = ClasswiseImageFolder(args.image_dir, transform=transform)

    exp_name = f"softvq_tbpostq_b{args.bits}_r{abs(args.range)}_n{args.images_per_class}"
    exp_dir  = os.path.join(args.output_dir, exp_name)
    real_dir = os.path.join(exp_dir, "real")
    recon_dir = os.path.join(exp_dir, "recon")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    # 3) 量化范围（与 TokenBridge 保持一致：全局 [-R, R]）
    qmin, qmax = -args.range, args.range
    stats = {"out_of_range_count": 0, "total_values": 0, "unique_values": set()}

    # 4) rFID 累计器
    rfid = RfidMeter(device=device)

    with torch.no_grad():
        for cls in tqdm(range(start_cls, end_cls), desc=f"GPU{rank} cls {start_cls}-{end_cls-1}"):
            idxs = dataset.samples_by_class.get(cls, [])[:args.images_per_class]
            if not idxs:
                continue
            loader = DataLoader(Subset(dataset, idxs),
                                batch_size=args.batch_size, num_workers=args.workers,
                                pin_memory=True, shuffle=False, drop_last=False)
            for images, _ in loader:
                images = images.to(device)

                # --- SoftVQ encode：拿连续 latent（建议用 quant，分布与解码器匹配） ---
                # 你的 VQModel.encode 返回：quant, emb_loss, info
                quant, _, _ = vq.encode(images)   # quant 形状：(B, L, D) 或 (B, C, H', W')

                lat = quant  # 不做 0.2325 缩放
                
                global_stats.update_values(lat, hist_min=qmin, hist_max=qmax)
                per_ch_stats.update_values(lat, hist_min=qmin, hist_max=qmax)
                
                # --- TokenBridge 式后量化 ---
                q_idx, lat_deq = post_training_quantize(lat, bits=args.bits, min_range=qmin, max_range=qmax)
                n_levels = 2 ** args.bits
                global_stats.update_indices(q_idx, n_levels=n_levels)
                per_ch_stats.update_indices(q_idx, n_levels=n_levels)
                
                # 统计 out-of-range（看原始 latent 被截断比例）
                stats["out_of_range_count"] += ((lat < qmin) | (lat > qmax)).sum().item()
                stats["total_values"] += lat.numel()
                stats["unique_values"].update(q_idx.detach().cpu().unique().tolist())

                # --- SoftVQ decode ---
                # VQModel.decode(quant, x=None, h=None, w=None)
                # 对 ViT 解码器不要求 h,w；CNN 解码器可能需要 h,w
                B, _, H, W = images.shape
                recon = vq.decode(lat_deq, x=images, h=H, w=W)  # 输出 [-1,1]

                # --- 保存 & 指标累加 ---
                rec_u8 = torch.clamp(127.5 * recon + 128, 0, 255).to(torch.uint8)
                img_u8 = torch.clamp(127.5 * images + 128, 0, 255).to(torch.uint8)

                rfid.update(img_u8, rec_u8)

                for i, (img_t, rec_t) in enumerate(zip(images, recon)):
                    save_image(img_t,  os.path.join(real_dir,  f"class_{cls:04d}_gpu{rank}_img_{i:02d}.png"))
                    save_image(rec_t,  os.path.join(recon_dir, f"class_{cls:04d}_gpu{rank}_img_{i:02d}.png"))

    report_global = global_stats.finalize()
    report_per_ch = per_ch_stats.finalize()
    exp_name = f"softvq_tbpostq_b{args.bits}_r{abs(args.range)}_n{args.images_per_class}"
    exp_dir  = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    out_report = {
        "shape_note": "ViT latent assumed as (B, L, D); per-channel stats along D",
        "bits": args.bits,
        "range": [-args.range, args.range],
        "global": report_global,
        "per_channel": report_per_ch
    }
    with open(os.path.join(exp_dir, f"latent_stats_rank{rank}.json"), "w") as f:
        json.dump(out_report, f, indent=2)    
    # 返回：重建图路径、统计、rFID 此进程值（若单机单卡则是最终值）
    return recon_dir, stats, rfid.compute()

def main():
    import argparse
    parser = argparse.ArgumentParser("Post-quantize SoftVQ latents (TokenBridge style) and eval rFID / IS")
    # SoftVQ / 数据
    parser.add_argument("--vq-model", type=str, default="./SOFTVQVAE/softvq")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--image-dir", type=str, default="ImageNet/val")
    # 量化
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--range", type=float, default=5.0)
    # 评测与存储
    parser.add_argument("--output-dir", type=str, default="quant_results_softvq")
    parser.add_argument("--images-per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    # 多进程
    parser.add_argument("--gpus", type=int, default=1, help="默认检测可用 GPU 数")
    args = parser.parse_args()

    world = args.gpus
    classes_per_gpu = max(1, 1000 // world)
    procs, per_stats, per_rfid = [], [], []

    for r in range(world):
        s = r * classes_per_gpu
        e = s + classes_per_gpu if r != world - 1 else 1000
        p = mp.Process(target=_worker, args=(r, args, s, e))
        p.start(); procs.append(p)

    for p in procs: p.join()

    # 汇总（简单起见：直接取输出目录计算 IS；rFID 可在单卡直接得到）
    exp_name = f"softvq_tbpostq_b{args.bits}_r{abs(args.range)}_n{args.images_per_class}"
    exp_dir  = os.path.join(args.output_dir, exp_name)
    recon_dir = os.path.join(exp_dir, "recon")

    print("\n[Compute IS from recon dir]")
    is_mean, is_std = compute_inception_score_from_dir(recon_dir)

    # rFID：若多进程严格聚合，可把 real/recon 都放一个目录由单个进程算；此处给出 recon-only 的 IS，以及
    #       建议你在单卡场景直接用 rfid.compute() 的输出作为最终 rFID。
    # 为了给出一个结果文件，我们把 IS 与参数持久化：
    results = {
        "parameters": vars(args),
        "inception_score": {"mean": is_mean, "std": is_std},
        "note": "rFID 已在各进程内 compute；若使用单卡运行，此值即最终 rFID。多卡严格聚合可在单进程统一计算。"
    }
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nIS: {is_mean:.4f} ± {is_std:.4f}")
    print(f"Saved to {os.path.join(exp_dir, 'results.json')}")

# 为了让 mp 调用更干净，单独包一层
def _worker(rank, args, s, e):
    torch.cuda.set_device(rank % max(1, torch.cuda.device_count()))
    exp_name = f"softvq_tbpostq_b{args.bits}_r{abs(args.range)}_n{args.images_per_class}"
    exp_dir  = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    recon_dir, stats, rfid_val = process_classes(rank, args, s, e)

    # 每个进程都各自写一个 stats json，方便你检查覆盖率/唯一值数等
    out = {
        "rank": rank,
        "class_range": [s, e],
        "out_of_range_ratio": (stats["out_of_range_count"] / max(1, stats["total_values"])),
        "unique_values_count": len(stats["unique_values"]),
        "rfid_local": rfid_val
    }
    with open(os.path.join(exp_dir, f"stats_rank{rank}.json"), "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
