import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import airfrans as af
import torch
import torch.nn.functional as F
from torch import nn
from enf.steerable_attention.equivariant_cross_attention_nef import EquivariantCrossAttentionNeF
from enf.steerable_attention.equivariant_cross_attention import RelativePositionND
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from utils.dataset import MultiElementSDFDataset
from utils.meta_learning import inner_loop
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import datetime

def train_encoder(cfg, train_dataset, val_dataset,  run_folder):
    # 1) Unpack config
    device        = torch.device(cfg.train.device)
    batch_size    = cfg.optim.batch_size
    epochs        = cfg.optim.epochs
    inner_steps   = cfg.optim.inner_steps
    lr_enf        = cfg.optim.lr_enf
    lr_latents    = cfg.optim.lr_latents
    num_points    = cfg.optim.num_points
    coord_dim     = cfg.model.coord_dim       # e.g. 2
    out_dim       = cfg.model.num_out    # e.g. 1

    # 2) Build ENF backbone
    enf = EquivariantCrossAttentionNeF(
        num_hidden=cfg.model.num_hidden,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        num_out=out_dim,
        latent_dim=cfg.model.latent_dim,
        cross_attn_invariant=RelativePositionND(coord_dim),
        self_attn_invariant=RelativePositionND(coord_dim),
        embedding_freq_multiplier=(cfg.model.emb_freq_q, cfg.model.emb_freq_v),
        condition_value_transform=True,
        use_gaussian_window=True,
        use_enhanced_stem=cfg.model.use_enhanced_stem,
    ).to(device)

    # Inner lr for latent features (positions and window frozen)
    alpha_c = nn.Parameter(
    torch.ones(cfg.model.latent_dim, device=device) * cfg.optim.lr_latents
    )

    optimizer_nef = torch.optim.AdamW([{'params': enf.parameters(),'lr': cfg.optim.lr_enf}])
    optimizer_inner_lrs = torch.optim.Adam([{'params': [alpha_c],'lr': 1e-4}])
    
    # Create a checkpoints subfolder inside run_folder
    ckpt_dir = os.path.join(run_folder, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val = float('inf')
    best_state = None

    # 3) Epoch loop
    for epoch in range(epochs):
        # --- Training (outer-loop update) ---
        enf.train()
        train_loss = 0.0
        train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
        for data in train_loader:
            flat_x = data.input    # [B*C, coord_dim]
            flat_y = data.output   # [B*C, out_dim]
            B = data.num_graphs
            C = flat_x.size(0) // B
            coords = flat_x.view(B, C, -1)    # -> [B, C, D]
            imgs   = flat_y.view(B, C, -1)    # -> [B, C, out_dim]

            # Inner-loop: fit c for this batch: is train True for 2nd order grad
            out, imgs_sub, _ = inner_loop(enf, cfg, coords, imgs, alpha_c, is_train=True)

            # Outer loss = MSE wrt ground truth
            loss = F.mse_loss(out, imgs_sub, reduction='mean')

            optimizer_nef.zero_grad()
            optimizer_inner_lrs.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enf.parameters(), max_norm=1.0)
            alpha_c.data.clamp_(1e-6, 10.0)
            optimizer_nef.step()
            optimizer_inner_lrs.step()

            train_loss += loss.item() * data.num_graphs

        train_loss /= len(train_dataset)

        # --- Validation (no outer update, only inner loop) ---
        enf.eval()
        val_loss = 0.0
        
        val_loader = DataLoader(val_dataset,batch_size= batch_size, shuffle=False)
        for data in val_loader:
            flat_x = data.input
            flat_y = data.output
            B = data.num_graphs
            C = flat_x.size(0) // B
            coords = flat_x.view(B, C, -1)
            imgs   = flat_y.view(B, C, -1)

            # Only inner loop
            out, imgs_sub, _ = inner_loop(enf, cfg, coords, imgs, alpha_c, is_train=False)
            loss = F.mse_loss(out, imgs_sub, reduction='mean')
            val_loss += loss.item() * data.num_graphs

        val_loss /= len(val_dataset)

        print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}, Inner lr: {alpha_c.mean():.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "avg_alpha": alpha_c.mean().item()
        })

        # Checkpoint best
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                'epoch': epoch,
                'model_state': enf.state_dict(),
                'alpha_c': alpha_c,
                'optimizer_state': optimizer_nef.state_dict()
            }
            torch.save(best_state, os.path.join(ckpt_dir, 'best_enf.pt'))

    return best_state, enf, alpha_c

def save_all_modulations(enf, alpha_c, cfg, dataset, device, out_path):
    """
    Iterate over `dataset` (which should be a torch_geometric dataset or list of data objects).
    For each batch/sample, run `inner_loop(...)` with is_train=False to get final p, c, g.
    Accumulate them (stack) into three big tensors:
      - all_p: shape [N_total, Z, coord_dim]
      - all_c: shape [N_total, Z, latent_dim]
      - all_g: shape [N_total, Z, 1]
    Then save a dictionary {"points": all_p, "latents": all_c, "gaussian": all_g} to disk via torch.save.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch_size=1 so we save per-sample easily
    all_p_list = []
    all_c_list = []
    all_g_list = []
    loss_list = []

    coord_dim   = cfg.model.coord_dim
    latent_dim  = cfg.model.latent_dim
    Z           = cfg.model.num_latents
    device      = torch.device(cfg.train.device)
    sample_size = cfg.optim.num_points

    enf.eval()
    for idx, data in enumerate(tqdm(loader, desc=f"Extracting modulations: {out_path}")):
        flat_x = data.input    # [C, coord_dim] since batch_size=1 => B=1, C = data.num_points
        flat_y = data.output   # [C, out_dim]
        B = data.num_graphs    # should be 1
        C = flat_x.size(0)

        coords = flat_x.view(1, C, -1).to(device)    # [1, C, coord_dim]
        imgs   = flat_y.view(1, C, -1).to(device)    # [1, C, out_dim]

        # Run inner_loop with is_train=False to fit c, but KEEP the final p, c, g
        out, imgs_sub, (p, c, g) = inner_loop(enf, cfg, coords, imgs, alpha_c, is_train=False)
        # Compute MSE on this sample’s subset:
        sample_loss = F.mse_loss(out, imgs_sub, reduction='mean').item()
        loss_list.append(sample_loss)
        
        # p: [1, Z, coord_dim], c: [1, Z, latent_dim], g: [1, Z, 1]
        # Compute CPU copy and squeeze batch‐dimension
        p_cpu = p.squeeze(0).cpu()    # [Z, coord_dim]
        c_cpu = c.squeeze(0).cpu()    # [Z, latent_dim]
        g_cpu = g.squeeze(0).cpu()    # [Z, 1]

        all_p_list.append(p_cpu)
        all_c_list.append(c_cpu)
        all_g_list.append(g_cpu)

    # Now stack along a new 0th dimension: N × Z × ...
    all_p = torch.stack(all_p_list, dim=0)  # [N_samples, Z, coord_dim]
    all_c = torch.stack(all_c_list, dim=0)  # [N_samples, Z, latent_dim]
    all_g = torch.stack(all_g_list, dim=0)  # [N_samples, Z, 1]
    
    # Compute average MSE over all samples:
    avg_loss = sum(loss_list) / len(loss_list)
    print(f"   ‣ Average reconstruction MSE: {avg_loss:.6f}")

    # Save as a dictionary
    to_save = {
        "points": all_p,
        "latents": all_c,
        "gaussian_window": all_g
    }
    torch.save(to_save, out_path)
    print(f"→ Saved modulations to {out_path} (N={len(all_p_list)}, Z={Z})")



@hydra.main(version_base=None, config_path=".", config_name="config_in")
def train(cfg: DictConfig):
    base_dir = cfg.logging.log_dir
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join(base_dir, now)
    os.makedirs(run_folder, exist_ok=True)
    print(f"=== Run output folder: {run_folder} ===")
    
    wandb.init(project=cfg.proj_name, config=OmegaConf.to_container(cfg))
    # Build datasets
    train_ds = MultiElementSDFDataset(
            database_path=cfg.dataset.root_path,
            is_train=True,
            num_points=None,  # Subsample to 500 points
            train_split=0.9,
            random_state=42
        )
        
    
    
    # Create test dataset using training normalization
    print("\n2. Creating test dataset...")
    val_ds = MultiElementSDFDataset(
        database_path=cfg.dataset.root_path,
        is_train=False,
        coef_norm=train_ds.coef_norm,  # Use training normalization
        num_points=None,
        train_split=0.9,
        random_state=42
    )

    # 4) Kick off training
    best_state, trained_enf, best_alpha_c = train_encoder(cfg, train_ds.processed_dataset, val_ds.processed_dataset, run_folder)
    print("Done training. Best epoch:", best_state['epoch'])

    
    meta = {
        "best_epoch": best_state['epoch'],
        "best_val_loss": float(best_val) if (best_val := best_state.get('val_loss', None)) else None
    }
    with open(os.path.join(run_folder, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    mods_folder = os.path.join(run_folder, "modulations")
    os.makedirs(mods_folder, exist_ok=True)

    train_mod_path = os.path.join(mods_folder, "train_modulations.pt")
    save_all_modulations(trained_enf, best_alpha_c, cfg, train_ds.processed_dataset, torch.device(cfg.train.device), train_mod_path)

    val_mod_path = os.path.join(mods_folder, "val_modulations.pt")
    save_all_modulations(trained_enf, best_alpha_c, cfg, val_ds.processed_dataset, torch.device(cfg.train.device), val_mod_path)

    print(f"All done! Modulations for train & val saved under:\n  {train_mod_path}\n  {val_mod_path}")
if __name__ == "__main__":
    train()