import torch
import torch.nn.functional as F

def initialise_latents(
    num_latents: int,
    batch_size: int,
    latent_dim: int,
    in_dims: int = 2,
    spatial_dims: int = 2,
    gaussian_width: float = -1.0,
    device: torch.device = torch.device("cpu"),
    latent_bbox = None,
):
    """
    Returns:
      p: [B, Z, coord_dim] grid of poses (either uniform in latent_bbox or symmetric [-lims,lims])
      c: [B, Z, latent_dim] learnable codes
      g: [B, Z, 1] fixed gaussian widths
    """
    
    spatial_dims = spatial_dims or in_dims
    # how many points per axis
    n = int(round(num_latents ** (1.0 / spatial_dims)))
    assert n**spatial_dims == num_latents, "num_latents must be a perfect power of coord_dim"
    
    # build each axis
    if latent_bbox is None:
        lims = 1.0 - 1.0 / n
        axis_ranges = [(-lims, lims)] * spatial_dims
    else:
        assert len(latent_bbox) == spatial_dims, "latent_bbox must have one (min,max) per dim"
        axis_ranges = latent_bbox

    axes = [
        torch.linspace(lo, hi, n, device=device)
        for lo, hi in axis_ranges
    ]
    # mesh and flatten
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)  # [n,...,n,coord_dim]
    grid = grid.reshape(-1, spatial_dims)                                # [Z,coord_dim]
    
    # If we have additional non-spatial dimensions, pad with zeros
    if in_dims > spatial_dims:
        zeros = torch.zeros((grid.shape[0], in_dims - spatial_dims), device=device)
        positions = torch.concatenate([grid, zeros], axis=-1)
    else:
        positions = grid

    # broadcast to batch
    p = positions.unsqueeze(0).repeat(batch_size, 1, 1)                    # [B,Z,coord_dim]

    # latents and widths
    c = torch.ones(batch_size, num_latents, latent_dim, device=device, requires_grad=True)
    if gaussian_width > 0:
        g = torch.full((batch_size, num_latents, 1), gaussian_width, device=device)
    else:
        g = torch.full((batch_size, num_latents, 1), 2.0 / n, device=device)

    return p, c, g


def inner_loop(enf, config, coords, imgs, alpha_c, is_train=True):
    """
    Solve per‐batch latents c via gradient descent, given fixed p, g.
    Returns final reconstruction + (p,c,g).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N, D     = coords.shape
    sample_size = config.optim.num_points if config.optim.num_points is not None else N
    Z = config.model.num_latents
    latent_dim = config.model.latent_dim
    num_steps = config.optim.inner_steps
    
    # init latents
    p, c, g = initialise_latents(Z, B, latent_dim, coords.size(-1), coords.size(-1), config.model.gaussian_window, device, latent_bbox=config.model.latent_bbox)
    p, g = p.to(device), g.to(device)
    c = c.to(device)

    for _ in range(num_steps):
        idx = torch.randperm(N)[:sample_size]
        # 2) index & then move to device
        x_sub = coords[:, idx, :].to(device)   # [B, S, D]
        y_sub = imgs[:,   idx, :].to(device)   # [B, S, out_dim]

        # 3) forward & grad wrt c
        out = enf(x_sub, p, c, g)               # [B, S, out_dim]
        loss = F.mse_loss(out, y_sub)
        grad_c, = torch.autograd.grad(loss, c, create_graph=is_train, retain_graph=is_train)

        # 4) update c with its per-dim alpha
        c = c - alpha_c.view(1,1,-1) * grad_c

    # final reconstruction on *one more fresh mask*
    idx = torch.randperm(N)[:sample_size]
    x_sub = coords[:, idx, :].to(device)
    y_sub = imgs[:,   idx, :].to(device)
    out   = enf(x_sub, p, c, g)
    return out, y_sub, (p, c, g)


