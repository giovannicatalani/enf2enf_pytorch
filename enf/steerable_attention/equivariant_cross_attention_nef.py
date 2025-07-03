# equivariant_cross_attention_nef.py
import torch
import torch.nn as nn
from functools import partial
from enf.steerable_attention.equivariant_cross_attention import RelativePositionND,EquivariantCrossAttention,LightweightSelfAttention, PointwiseFFN



class LightweightSelfAttentionBlock(nn.Module):
    """
    Self-attention block with residual connection and normalization.
    """
    def __init__(self, num_hidden: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(num_hidden)
        self.self_attn = LightweightSelfAttention(num_hidden, num_heads, dropout)
        self.ffn = PointwiseFFN(num_hidden, num_hidden, num_hidden)
        
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: [B, Z, H] latent features
        Returns:
            [B, Z, H] processed latent features
        """
        # Self-attention with residual
        a_norm = self.ln(a)
        a_attn = self.self_attn(a_norm)
        a = a + a_attn
        
        # FFN with residual
        a = a + self.ffn(self.ln(a))
        
        return a

class EquivariantCrossAttentionBlock(nn.Module):
    def __init__(self, num_hidden, num_heads, attn_factory, residual, project_heads):
        super().__init__()
        self.residual = residual
        # <--- always normalize over H, not heads*H
        self.ln = nn.LayerNorm(num_hidden)
        self.attn = attn_factory()
        in_dim = num_hidden if project_heads else num_heads * num_hidden
        self.ffn = PointwiseFFN(in_dim, in_dim, in_dim)

    def forward(self, x, p, a, x_h, window_sigma, *, return_attn=False):
        # a: [B, Z, H]
        a_norm = self.ln(a)
        if return_attn:
            a_attn, att = self.attn(x, p, a_norm, window_sigma, x_h, return_attn=return_attn)
        else:
            a_attn = self.attn(x, p, a_norm, window_sigma, x_h)
            
        out = self.ffn(a + a_attn) if self.residual else self.ffn(a_attn)
        
        return (out, att) if return_attn else out



class EquivariantCrossAttentionNeF(nn.Module):
    """
    Stacks latent self‐attention blocks, then a final cross‐attention to map
    code latents → solution values at query points.

    Args:
        num_hidden, num_heads, num_layers, num_out, latent_dim: as in JAX
        cross_attn_invariant, self_attn_invariant: e.g. RelativePositionND(...)
        embedding_freq_multiplier: tuple(freq_q, freq_v)
        condition_value_transform: bool
        use_gaussian_window: bool
        use_enhanced_stem: bool
    """
    def __init__(self,
                 num_hidden, num_heads, num_layers, num_out, latent_dim,
                 cross_attn_invariant, self_attn_invariant,
                 embedding_freq_multiplier, condition_value_transform,
                 use_gaussian_window=True, use_enhanced_stem=False, use_lightweight_self_attn=False):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_lightweight_self_attn = use_lightweight_self_attn

        # Stem from latent_dim → H
        if use_enhanced_stem:
            self.latent_stem = nn.Sequential(
                nn.Linear(latent_dim, num_hidden), nn.GELU(),
                nn.Linear(num_hidden, num_hidden)
            )
        else:
            self.latent_stem = nn.Linear(latent_dim, num_hidden)

        # Factories for attention blocks
        cross_factory = partial(
            EquivariantCrossAttention,
            num_hidden=num_hidden,
            num_heads=num_heads,
            invariant=cross_attn_invariant,
            embedding_freq_multiplier=embedding_freq_multiplier,
            condition_value_transform=condition_value_transform,
            condition_invariant_embedding=False,
            project_heads=False,
            use_gaussian_window=use_gaussian_window
        )
# Choose self-attention type
        if use_lightweight_self_attn:
            # Lightweight self-attention blocks
            self.self_blocks = nn.ModuleList([
                LightweightSelfAttentionBlock(num_hidden, num_heads)
                for _ in range(num_layers)
            ])
        else:
            # Original heavy equivariant self-attention
            self_factory = partial(
                EquivariantCrossAttention,
                num_hidden=num_hidden,
                num_heads=num_heads,
                invariant=self_attn_invariant,
                embedding_freq_multiplier=embedding_freq_multiplier,
                condition_value_transform=condition_value_transform,
                condition_invariant_embedding=False,
                project_heads=True,
                use_gaussian_window=use_gaussian_window
            )
            self.self_blocks = nn.ModuleList([
                EquivariantCrossAttentionBlock(
                    num_hidden, num_heads, self_factory,
                    residual=True, project_heads=True
                )
                for _ in range(num_layers)
            ])
            
        # Final cross‐attn block
        self.cross_blocks = nn.ModuleList([
            EquivariantCrossAttentionBlock(
                num_hidden, num_heads, cross_factory,
                residual=False, project_heads=False
            )
        ])

        # FINAL MLP HEAD must accept num_heads * num_hidden
        self.out_proj = nn.Sequential(
            nn.Linear(num_heads * num_hidden, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_out),
        )
        self.activation = nn.GELU()

    def forward(self, x, p, a, window_sigma, *, return_attn=False):
        a = self.latent_stem(a)  # [B, Z, H]

        if self.use_lightweight_self_attn:
            # Lightweight self-attention (no spatial reasoning)
            for block in self.self_blocks:
                a = block(a)  # Much simpler interface!
        else:
            # Original heavy self-attention
            for block in self.self_blocks:
                attn_out = block(p, p, a, None, window_sigma)
                a = a + attn_out
                a = self.activation(a)

        # final cross‐attention
        if return_attn:
            out, att = self.cross_blocks[0](x, p, a, None, window_sigma,return_attn=return_attn)
        else:
            out = self.cross_blocks[0](x, p, a, None, window_sigma)
            
        out = self.out_proj(self.activation(out))

        # PROJECT from heads*H → H → … → num_out
        return (out, att) if return_attn else out

if __name__ == '__main__':
    
    # Hyper-parameters
    B, C, D = 3, 100, 2       # batch, #coords, coord-dim
    Z, latent_dim = 9, 8    # #latents, raw latent dim
    H, num_out, num_layers, num_heads = 256, 4, 2, 2

    # Build model
    inv = RelativePositionND(D)
    nef = EquivariantCrossAttentionNeF(
        num_hidden=H,
        num_heads=num_heads,
        num_layers=num_layers,
        num_out=num_out,
        latent_dim=latent_dim,
        cross_attn_invariant=inv,
        self_attn_invariant=inv,
        embedding_freq_multiplier=(1.0, 1.0),
        condition_value_transform=True,
        use_gaussian_window=True,
        use_enhanced_stem=True,
    )

    # Dummy inputs
    x = torch.randn(B, C, D)                  # query coordinates
    p = torch.randn(B, Z, D)                  # latent poses
    a_raw = torch.randn(B, Z, latent_dim)     # raw latents
    window_sigma = torch.ones(B, Z, 1)        # gaussian widths

    y = nef(x, p, a_raw, window_sigma)  # forward pass)
    print("Final output shape:", y.shape)          # (B, C, num_out)
