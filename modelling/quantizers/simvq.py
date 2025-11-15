from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class simVectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,   # number of codewords (K)
        token_size:     int = 256,   # embedding dim (D)
        commitment_cost: float = 0.25,
        use_l2_norm:     bool = False,
        simvq:           bool = True,  # freeze codebook, learn projection W
        entropy_loss_ratio: float = 0.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_size = token_size
        self.commitment_cost = commitment_cost
        self.use_l2_norm = use_l2_norm
        self.simvq = simvq
        self.entropy_loss_ratio = entropy_loss_ratio

        # Codebook C: [K, D]
        self.embedding = nn.Embedding(codebook_size, token_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.token_size ** -0.5)

        # SimVQ: freeze C and learn linear projection W: R^D -> R^D
        if self.simvq:
            for p in self.embedding.parameters():
                p.requires_grad = False
            self.embedding_proj = nn.Linear(token_size, token_size)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, L, D] ViT token features
        Returns:
            z_q: [B, L, D] quantized tokens (STE applied)
            (codebook_loss, commitment_loss, entropy_loss, usage(not logged))
            info: (None, None, indices[B,1,L], 1., 1., z_q_detach[B,L,D], z_detach[B,L,D], cos_mean)
        """
        if z.dim() != 3:
            raise ValueError(f"simVQ expects 3D [B,L,D] inputs (ViT path only), got {tuple(z.shape)}")
        z = z.float()
        B, L, D = z.shape
        device  = z.device

        # Project codebook if SimVQ mode
        weight = self.embedding.weight  # [K, D]
        if self.simvq:
            weight = self.embedding_proj(weight)  # [K, D]

        if self.use_l2_norm:
            z_for_dist = F.normalize(z,      dim=-1, eps=1e-6)  # [B,L,D]
            w_for_dist = F.normalize(weight, dim=-1, eps=1e-6)  # [K,D]
        else:
            z_for_dist = z
            w_for_dist = weight

        # Squared Euclidean distances: ||z||^2 + ||w||^2 - 2 z·w
        # z2: [B,L,1], w2: [1,1,K], dot: [B,L,K]
        z2   = (z_for_dist ** 2).sum(dim=-1, keepdim=True)
        w2   = (w_for_dist ** 2).sum(dim=-1).view(1, 1, -1)
        dot  = torch.einsum('bld,kd->blk', z_for_dist, w_for_dist)
        dists = z2 + w2 - 2 * dot                              # [B,L,K]

        # Hard NN assignment
        indices = torch.argmin(dists, dim=-1)                  # [B, L]
        q = F.embedding(indices, w_for_dist if self.use_l2_norm else weight)                       # [B, L, D]

        # Losses
        # commitment: beta * ||q.detach - z||^2 ; codebook: ||q - z.detach||^2
        z_cmp = F.normalize(z, dim=-1, eps=1e-6) if self.use_l2_norm else z
        commitment_loss = self.commitment_cost * ((q.detach() - z_cmp) ** 2).mean()
        codebook_loss   = ((q - z_cmp.detach()) ** 2).mean()

        # Straight-through estimator
        z_q = z + (q - z).detach()                             # [B, L, D]

        # For logging/compatibility with SoftVQ info
        with torch.no_grad():
            cos_mean = F.cosine_similarity(z.reshape(-1, D), z_q.reshape(-1, D), dim=-1).mean()

        entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(dot, temperature=0.01)
        # Assemble outputs expected by training/VQLoss
        usage        = torch.tensor(0., device=device)         # not tracked; keep zero
        indices_3d   = indices.unsqueeze(1)                    # [B, 1, L]
        probs = F.softmax(dot, dim=-1)        # [B,L,K]
        avg_probs = probs.mean()
        max_probs = probs.max(dim=-1)[0].mean()

        info = (
            None,              # perplexity placeholder
            None,              # min_encodings placeholder
            indices_3d,        # [B,1,L]
            avg_probs,
            max_probs,
            z_q.detach(),      # [B,L,D]
            z.detach(),        # [B,L,D]
            cos_mean
        )
        return z_q, (codebook_loss, commitment_loss, entropy_loss, usage), info

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def get_codebook_entry(self, code_b: torch.Tensor, shape=None,channel_first=True):
        """
        Map indices to embeddings for decode_code.
        Accepts [B,L] or [B,1,L] or flat [N] (then provide shape=(B,L) or (B,H,W)).
        Returns [B, L, D] for ViT path.
        channel_first: legacy
        """
        weight = self.embedding.weight
        if self.simvq:
            weight = self.embedding_proj(weight)
        if self.use_l2_norm:
            weight = F.normalize(weight, dim=-1, eps=1e-6)

        # Normalize indices to [B, L]
        if code_b.dim() == 3:              # [B,1,L] or [B,H,W] already flattened via view
            B = code_b.size(0)
            code_b = code_b.view(B, -1)
        elif code_b.dim() == 2:            # [B,L]
            B = code_b.size(0)
        elif code_b.dim() == 1:            # [N] -> need shape
            if shape is None:
                raise ValueError("When code_b is 1D, provide shape=(B,L) or (B,H,W) for ViT path.")
            if len(shape) == 2:
                B, L = shape
            elif len(shape) == 3:
                B, H, W = shape
                L = H * W
            else:
                raise ValueError(f"Unsupported shape: {shape}")
            code_b = code_b.view(B, -1)
        else:
            raise ValueError(f"Unsupported indices shape: {tuple(code_b.shape)}")

        flat = code_b.reshape(-1).long()            # [B*L]
        feats = F.embedding(flat, weight)           # [B*L, D]
        B, L = code_b.shape
        D    = feats.size(-1)
        feats = feats.view(B, L, D)                 # [B, L, D]
        return feats

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
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss
