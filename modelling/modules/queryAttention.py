import torch
import torch.nn as nn
from typing import Optional, Tuple, List

class FFN(nn.Module):
    """Pre-LN Transformer FFN: Linear -> GELU -> Dropout -> Linear."""
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.0):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QueryAttnLayer(nn.Module):
    """
      x = Query; c = Context
      y = x + CrossAttn(LN(x), c)
      z = y + SelfAttn(LN(y))
      o = z + FFN(LN(z))
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        d_ff = d_ff or (4 * d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout, batch_first=True
        )
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,                 # (B, Nq, D)
        context: torch.Tensor,               # (B, Nc, D)
        query_key_padding_mask: Optional[torch.Tensor] = None,   # (B, Nq), True=pad
        context_key_padding_mask: Optional[torch.Tensor] = None, # (B, Nc), True=pad
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1) Cross-Attention: Q from query，K/V from context
        q1 = self.norm1(query)
        cross_out, cross_w = self.cross_attn(
            q1, context, context,
            key_padding_mask=context_key_padding_mask,
            need_weights=need_weights, attn_mask=None
        )
        y = query + self.drop1(cross_out)

        # 2) Self-Attention on queries
        q2 = self.norm2(y)
        self_out, _ = self.self_attn(
            q2, q2, q2,
            key_padding_mask=query_key_padding_mask,
            need_weights=False, attn_mask=None
        )
        z = y + self.drop2(self_out)

        # 3) FFN
        q3 = self.norm3(z)
        o = z + self.drop3(self.ffn(q3))

        return (o, cross_w if need_weights else None)

class QueryAttnModule(nn.Module):
    """
    query attention module to aggregate information from learned context
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int = 1,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            QueryAttnLayer(d_model, n_heads, d_ff=d_ff, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        query: torch.Tensor,                 # (B, Nq, D)
        context: torch.Tensor,               # (B, Nc, D)
        query_key_padding_mask: Optional[torch.Tensor] = None,   # (B, Nq)
        context_key_padding_mask: Optional[torch.Tensor] = None, # (B, Nc)
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        attns = [] if need_weights else None
        x = query
        for layer in self.layers:
            x, w = layer(
                x, context,
                query_key_padding_mask=query_key_padding_mask,
                context_key_padding_mask=context_key_padding_mask,
                need_weights=need_weights,
            )
            if need_weights:
                attns.append(w)  # cross-attn weigths (B, heads, Nq, Nc)
        return (x, attns)
