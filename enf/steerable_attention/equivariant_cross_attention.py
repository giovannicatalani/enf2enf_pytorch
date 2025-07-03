import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightSelfAttention(nn.Module):
    """
    Simple, efficient self-attention for latent-to-latent communication.
    No spatial reasoning, just feature mixing.
    """
    def __init__(self, num_hidden: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.head_dim = num_hidden // num_heads
        
        assert num_hidden % num_heads == 0, "num_hidden must be divisible by num_heads"
        
        # Standard QKV projections
        self.q_proj = nn.Linear(num_hidden, num_hidden)
        self.k_proj = nn.Linear(num_hidden, num_hidden)
        self.v_proj = nn.Linear(num_hidden, num_hidden)
        self.out_proj = nn.Linear(num_hidden, num_hidden)
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: [B, Z, H] latent features
        Returns:
            [B, Z, H] attended latent features
        """
        B, Z, H = a.shape
        
        # Project to Q, K, V
        q = self.q_proj(a)  # [B, Z, H]
        k = self.k_proj(a)  # [B, Z, H]
        v = self.v_proj(a)  # [B, Z, H]
        
        # Reshape for multi-head attention: [B, Z, num_heads, head_dim]
        q = q.view(B, Z, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, Z, head_dim]
        k = k.view(B, Z, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, Z, head_dim]
        v = v.view(B, Z, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, Z, head_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, Z, Z]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, num_heads, Z, head_dim]
        
        # Reshape back: [B, Z, H]
        out = out.transpose(1, 2).contiguous().view(B, Z, H)
        
        # Final projection
        return self.out_proj(out)


class RelativePositionND(nn.Module):
    """ Calculate the relative position between two sets of coordinates in N dimensions.

        Args:
            num_dims (int): The dimensionality of the coordinates, corresponds 
                            to the dimensionality of the translation group.
    """

    def __init__(self, num_dims: int):
        super().__init__()

        # Set the dimensionality of the invariant.
        self.dim = num_dims

        # This invariant is calculated based on two sets of positional coordinates, it doesn't depend on
        # the orientation.
        self.num_x_pos_dims = num_dims
        self.num_x_ori_dims = 0
        self.num_z_pos_dims = num_dims
        self.num_z_ori_dims = 0

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, num_x_pos_dims]
        p: [B, Z, num_z_pos_dims]
        returns: [B, C, Z, num_x_pos_dims]
        """
        return x[:, :, None, :self.num_x_pos_dims] - p[:, None, :, :self.num_z_pos_dims]
    
    def calculate_gaussian_window(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Non-periodic Gaussian window:
          - squared distance between x and p, then scaled by 1/sigma^2 and negated.

        x:     [B, C, D]
        p:     [B, Z, D]
        sigma: [B, Z, 1]

        returns: [B, C, Z, 1]
        """
        # extract only the positional dims
        x_pos = x[..., :self.num_x_pos_dims]  # [B, C, D]
        p_pos = p[..., :self.num_z_pos_dims]  # [B, Z, D]

        # pairwise squared distances: (p_pos - x_pos)^2 summed over D
        # p_pos[:,None,:,:] -> [B, 1, Z, D]
        # x_pos[:,:,None,:] -> [B, C, 1, D]
        diffs = p_pos[:, None, :, :] - x_pos[:, :, None, :]
        sq_dists = (diffs ** 2).sum(dim=-1, keepdim=True)  # [B, C, Z, 1]

        # scale by -1 / sigma^2  (sigma[:,None,:] -> [B,1,Z,1], broadcasts)
        return - sq_dists / (sigma[:, None, :] ** 2)
    


class RFFEmbedding(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, learnable_coefficients: bool, std: float):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be even"
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.learnable = learnable_coefficients
        # weight shape [in_dim, hidden_dim//2]
        coeff = torch.randn(in_dim, hidden_dim // 2) * std
        self.coefficients = nn.Parameter(coeff, requires_grad=learnable_coefficients)
        self.pi = 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]
        # optionally detach coefficients if not learnable
        coeff = self.coefficients if self.learnable else self.coefficients.detach()
        # project
        # [..., in_dim] @ [in_dim, hidden_dim//2] -> [..., hidden_dim//2]
        x_proj = self.pi * x @ coeff
        # sin+cos concat
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class RFFNet(nn.Module):
    """RFF-based embedding network: encoding + MLP + final linear."""
    def __init__(
        self,
        in_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        learnable_coefficients: bool,
        std: float
    ):
        super().__init__()
        assert num_layers >= 2, "Need >=2 layers"
        # embedding
        self.encoding = RFFEmbedding(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            learnable_coefficients=learnable_coefficients,
            std=std
        )
        # hidden MLP layers
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        # final linear
        self.linear_final = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]
        x = self.encoding(x)
        x = self.mlp(x)
        return self.linear_final(x)


class PointwiseFFN(nn.Module):
    """Two-layer FFN with GELU and LayerNorm."""
    def __init__(self, num_in: int, num_hidden: int, num_out: int):
        super().__init__()
        self.fc1 = nn.Linear(num_in, num_hidden)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.ln(x)
        return self.fc2(x)


class EquivariantCrossAttention(nn.Module):
    """Equivariant cross-attention as in JAX/Flax version."""
    def __init__(
        self,
        num_hidden: int,
        num_heads: int,
        invariant: RelativePositionND,
        embedding_freq_multiplier: tuple,
        condition_value_transform: bool = False,
        condition_invariant_embedding: bool = False,
        project_heads: bool = True,
        use_gaussian_window: bool = True
    ):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.invariant = invariant
        inv_freq, val_freq = embedding_freq_multiplier
        # embeddings
        self.invariant_embedding_query = RFFNet(
            in_dim=invariant.dim,
            output_dim=num_hidden,
            hidden_dim=num_hidden,
            num_layers=2,
            learnable_coefficients=False,
            std=inv_freq
        )
        self.invariant_embedding_value = RFFNet(
            in_dim=invariant.dim,
            output_dim=num_hidden,
            hidden_dim=num_hidden,
            num_layers=2,
            learnable_coefficients=False,
            std=val_freq
        )
        # linear transforms
        self.inv_emb_to_q = nn.Linear(num_hidden, num_heads * num_hidden)
        self.a_to_k = nn.Linear(num_hidden, num_heads * num_hidden)
        self.a_to_v = nn.Linear(num_hidden, num_heads * num_hidden)
        # optional conditioning
        self.condition_value_transform = condition_value_transform
        self.condition_invariant_embedding = condition_invariant_embedding
        if condition_invariant_embedding:
            self.inv_emb_cond_to_inv_emb = PointwiseFFN(num_hidden, num_hidden, 2 * num_hidden)
        if condition_value_transform:
            self.inv_emb_to_v = PointwiseFFN(num_hidden, num_hidden, 2 * num_heads * num_hidden)
            self.inv_emb_cond_mixer = PointwiseFFN(num_hidden, num_hidden, num_hidden)
        # output projection
        out_dim = num_hidden if project_heads else num_heads * num_hidden
        self.out_proj = nn.Linear(num_heads * num_hidden, out_dim)
        # parameters
        self.scale = 1.0 / math.sqrt(num_hidden)
        self.use_gaussian_window = use_gaussian_window

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        a: torch.Tensor,
        window_sigma: torch.Tensor = None,
        x_h: torch.Tensor = None,
        *, return_attn=False
    ) -> torch.Tensor:
        """
        x: [B, C, D], p: [B, Z, D], a: [B, Z, H], window_sigma: [B, Z, 1]
        returns y: [B, C, out_dim]
        """
        # compute invariant [B, C, Z, dim]
        inv = self.invariant(x, p)
        # embed for query
        inv_emb_q = self.invariant_embedding_query(inv)
        q = self.inv_emb_to_q(inv_emb_q).view(*inv_emb_q.shape[:3], self.num_heads, self.num_hidden)
        # keys & values from a
        k = self.a_to_k(a).view(a.size(0), a.size(1), self.num_heads, self.num_hidden)
        v = self.a_to_v(a)  # will reshape later
        # condition value if needed
        if self.condition_value_transform:
            inv_emb_v = self.invariant_embedding_value(inv)
            if self.condition_invariant_embedding:
                assert x_h is not None
                gamma_beta = self.inv_emb_cond_to_inv_emb(x_h)  # [B,C,2H]
                gamma, beta = gamma_beta.chunk(2, dim=-1)
                inv_emb_v = inv_emb_v * (1 + gamma.unsqueeze(2)) + beta.unsqueeze(2)
            v_gb = self.inv_emb_to_v(inv_emb_v)  # [B,C,Z,2*H* heads]
            v_gamma, v_beta = v_gb.chunk(2, dim=-1)
            v = v.unsqueeze(1) * (1 + v_gamma) + v_beta
            v = v.view(*v.shape[:3], self.num_heads, self.num_hidden)
            v = self.inv_emb_cond_mixer(v)
        else:
            v = v.unsqueeze(1).view(*v.shape[:2], 1, self.num_heads, self.num_hidden)
            v = v.expand(-1, -1, inv.shape[2], -1, -1)
        # prepare k for broadcasting
        k = k.unsqueeze(1).expand(-1, x.size(1), -1, -1, -1)  # [B,C,Z,heads, H]
        # compute attention logits
        att_logits = (q * k).sum(dim=-1) * self.scale  # [B,C,Z,heads]
        if self.use_gaussian_window:
            # gaussian: [B,C,Z,1]
            p_pos = p[:, :, :self.invariant.num_z_pos_dims]
            x_pos = x[:, :, :self.invariant.num_x_pos_dims]
            # squared dist
            d2 = (p_pos.unsqueeze(1) - x_pos.unsqueeze(2)).pow(2).sum(-1, keepdim=True)
            att_logits = att_logits - d2 / (window_sigma.unsqueeze(1) ** 2)
        att = F.softmax(att_logits, dim=2)
        # attend to values
        y = (att.unsqueeze(-1) * v).sum(dim=2)  # [B,C,heads,H]
        # reshape & project
        y = y.view(y.size(0), y.size(1), -1)
        return (self.out_proj(y), att) if return_attn else self.out_proj(y) 

if __name__ == '__main__':
    # create dummy inputs
    batch_size  = 3
    seq_len     = 100
    coord_dim   = 2
    num_latents = 4
    num_hidden  = 256
    latent_dim = 2

    x            = torch.randn(batch_size, seq_len, coord_dim)
    p            = torch.randn(batch_size, num_latents, coord_dim)
    a            = torch.randn(batch_size, num_latents, num_hidden)
    window_sigma = torch.ones(batch_size, num_latents, 1)
    x_h          = torch.randn(batch_size, seq_len, num_hidden)

    # instantiate your module
    invariant = RelativePositionND(coord_dim)
    m = EquivariantCrossAttention(
        num_hidden=num_hidden,
        num_heads=2,
        invariant=invariant,
        embedding_freq_multiplier=(0.5, 10),
        condition_value_transform=True,
        condition_invariant_embedding=True,
        project_heads=True,
        use_gaussian_window=True
    )

    
