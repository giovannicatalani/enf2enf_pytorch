import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import airfrans as af
import torch
import torch.nn.functional as F
from torch import nn
from enf.steerable_attention.equivariant_cross_attention_nef import EquivariantCrossAttentionNeF
from enf.steerable_attention.equivariant_cross_attention import RelativePositionND
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from utils.dataset import ElasticityDataset
from utils.meta_learning import initialise_latents
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import yaml
import datetime
import time
import matplotlib.pyplot as plt


@hydra.main(config_path="", config_name="config_out.yaml")
def main(cfg: DictConfig) -> None:
    # Initialize wandb
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.init(project='elasticity_enf_torch', config=OmegaConf.to_container(cfg, resolve=True))
    wandb.config.update({
        "batch_size": cfg.optim.batch_size,
        "lr_enf": cfg.optim.lr_enf,
        "epochs": cfg.optim.epochs,
        "num_points": cfg.optim.num_points,
    })

    torch.set_default_dtype(torch.float32)

    # optim
    batch_size = cfg.optim.batch_size
    lr_enf = cfg.optim.lr_enf
    epochs = cfg.optim.epochs
    num_points = cfg.optim.num_points
    coord_dim = cfg.model.spatial_dim     
    out_dim = cfg.model.num_out  
    in_dim = cfg.model.in_dim
    latent_dim = cfg.model.latent_dim
    num_latents = cfg.model.num_latents
    latent_cond = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2) Build ENF backbone
    enf = EquivariantCrossAttentionNeF(
        num_hidden=cfg.model.num_hidden,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        num_out=out_dim,
        latent_dim=latent_dim,
        cross_attn_invariant=RelativePositionND(in_dim),
        self_attn_invariant=RelativePositionND(in_dim),
        embedding_freq_multiplier=(cfg.model.emb_freq_q, cfg.model.emb_freq_v),
        condition_value_transform=True,
        use_gaussian_window=True,
        use_enhanced_stem=cfg.model.use_enhanced_stem,
        use_lightweight_self_attn=cfg.model.use_lightweight_self_attn,
    ).to(device)
    
    train_latents = torch.load(cfg.dataset.train_latents_path)['latents'].detach().cpu()
    test_latents = torch.load(cfg.dataset.test_latents_path)['latents'].detach().cpu()
    latent_array = torch.cat((train_latents, test_latents), dim=0)
    train_dataset = ElasticityDataset(
        data_directory='/scratch/dmsm/gi.catalani/Elasticity/',
        latents=latent_array,  # Shape: [total_samples, num_latent_points, latent_dim]
        target_field='sigma',
        use_augmented_inputs=cfg.dataset.augmented_inputs,
        mode='train',
        ntrain=1000,
        ntest=200
        )

    # Get normalization for val/test
    norm_stats = train_dataset.get_normalization_stats()

    val_dataset = ElasticityDataset(
        data_directory='/scratch/dmsm/gi.catalani/Elasticity/', 
        latents=latent_array,
        target_field='sigma',
        use_augmented_inputs=cfg.dataset.augmented_inputs,
        mode='val',
        coef_norm=norm_stats,
        ntrain=1000,
        ntest=200
    )
    

    ntrain = len(train_dataset)
    nval = len(val_dataset)
    
    optimizer_nef = torch.optim.Adam([{'params': enf.parameters(),'lr': cfg.optim.lr_enf}])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_nef,
        mode='min',
        factor=0.99,      # Reduce LR when plateauing
        patience=50,     # Number of epochs to wait before reducing LR
        verbose=True,    # Print message when LR is reduced
        min_lr=1e-6    # Minimum LR threshold
    )
    
    
    # Improved folder naming with timestamp and all output target fields
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    folder_name = f"training_result_{timestamp}" 
    dir = os.path.dirname(__file__)
    results_directory = os.path.join(dir + '/experiments/training/', folder_name)
    os.makedirs(results_directory, exist_ok=True)
    cfg_save_path = os.path.join(results_directory, 'config.yaml')
    # Convert OmegaConf DictConfig to a native Python dictionary
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # Set `resolve=True` to resolve all interpolations
    with open(cfg_save_path, 'w') as file:
        yaml.dump(cfg_dict, file, default_flow_style=False)

    # Generate run name
    run_name = f"enf_out"
   
    best_loss = np.inf
    train_loss_history = []
    test_loss_history = []

  
    start_epoch = 0
    start_time = time.time()
    for step in tqdm(range(start_epoch, epochs)):
        fit_train_mse_in = 0
        fit_test_mse_in = 0
        fit_test_l2_rel_error = 0.0
        fit_field  = 0.0
        fit_test_field = 0.0
        test_loss_in  = 1
        # Initialize lists for loss history
        
        use_pred_loss = step % 5 == 0
        
        # Prepare the training dataset for the new epoch
        # Subsample the training dataset
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        print('Num points for training', len(train_dataset[0].input))
        for substep,graph in enumerate(train_loader): 

            n_samples = len(graph)
            enf.train()
            graph = graph.cuda() if torch.cuda.is_available() else graph
            
            # Training step implementation inline
            B = graph.num_graphs
            C = graph.input.size(0) // B
            Z = graph.cond.size(1)
            latent_dim = graph.cond.size(2)
            coords = graph.input.view(B, C, -1)
            features = graph.output.view(B, C, -1)

            p, _, g = initialise_latents(Z, B, latent_dim, coords.size(-1), cfg.model.spatial_dim,cfg.model.gaussian_window, device, latent_bbox=cfg.model.latent_bbox)
            p, g = p.to(device), g.to(device)
            c = graph.cond.to(device)

            with torch.set_grad_enabled(True):
                features_recon = enf(coords, p, c, g)  # [B, S, out_dim]
                loss = F.mse_loss(features_recon, features)

            optimizer_nef.zero_grad()
            loss.backward(create_graph=False)
            nn.utils.clip_grad_value_(enf.parameters(), clip_value=1.0)
            optimizer_nef.step()
            loss_value = loss.cpu().detach()
            fit_train_mse_in += loss_value.item() * n_samples

        
        train_loss_in = fit_train_mse_in / (ntrain)
        train_loss_history.append({'epoch': step, 'loss': train_loss_in})
        # Inside training loop
        wandb.log({"train_loss": train_loss_in, "epoch": step})
        print('Train Loss', train_loss_in)
       

        if use_pred_loss:
            # Prepare the training dataset for the new epoch
            
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            for substep,graph in enumerate(val_loader):  
                n_samples = len(graph)
                enf.eval()
                graph = graph.cuda() if torch.cuda.is_available() else graph
                
                # Training step implementation inline
                with torch.no_grad():
                    B = graph.num_graphs
                    C = graph.input.size(0) // B
                    Z = graph.cond.size(1)
                    latent_dim = graph.cond.size(2)
                    coords = graph.input.view(B, C, -1)
                    features = graph.output.view(B, C, -1)

                    p, _, g = initialise_latents(Z, B, latent_dim, coords.size(-1), cfg.model.spatial_dim, cfg.model.gaussian_window, device, latent_bbox=cfg.model.latent_bbox)
                    p, g = p.to(device), g.to(device)
                    c = graph.cond.to(device)                   
                    features_recon = enf(coords, p, c, g)  # [B, S, out_dim]
                    loss = F.mse_loss(features_recon, features)
                    
                    l2_rel_error = torch.norm(features_recon - features, dim=(1,2)) / torch.norm(features, dim=(1,2))
                    l2_rel_error = torch.mean(l2_rel_error)  # Average over batch

                        
                loss_val = loss.cpu().detach()
                l2_rel_error_val = l2_rel_error.cpu().detach()
        
                fit_test_mse_in += loss_val.item() * n_samples
                fit_test_l2_rel_error += l2_rel_error_val.item() * n_samples
        
            test_loss_in = fit_test_mse_in / (nval)
            test_l2_rel_error = fit_test_l2_rel_error / (nval)
            
            scheduler.step(test_loss_in)
            
            test_loss_history.append({'epoch': step, 'loss': test_loss_in, 'l2_rel_error': test_l2_rel_error})
            wandb.log({"val_loss": test_loss_in, "val_l2_rel_error": test_l2_rel_error, "epoch": step})
            print('Test Loss', test_loss_in)
            print('Test L2 Relative Error', test_l2_rel_error)

            plt.figure()
            plt.plot([p['epoch'] for p in train_loss_history], [p['loss'] for p in train_loss_history], label='Train Loss')
            plt.plot([p['epoch'] for p in test_loss_history], [p['loss'] for p in test_loss_history], label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.title('Training and Test Loss')
            plt.legend()
            plt.savefig(os.path.join(results_directory, f"{run_name}_loss_plot.png"))
            plt.close()
        
        if test_loss_in < best_loss:
            best_loss = test_loss_in
            
            torch.save(
                {
                    "cfg": cfg,
                    "epoch": step,
                    "enf": enf.state_dict(),
                    "optimizer_enf": optimizer_nef.state_dict(),
                },
                os.path.join(results_directory, f"{run_name}.pt"),
            )
            
            if cfg.logging.save_predictions:
                #Infer on full resolution dataset
                full_field_loss = 0.0
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
                all_predictions = []
                all_coords = []
                all_sample_indices = []
                for substep,graph in enumerate(val_loader):  
                    n_samples = len(graph)
                    enf.eval()
                    graph = graph.cuda() if torch.cuda.is_available() else graph
                    
                    # Training step implementation inline
                    with torch.no_grad():
                        B = graph.num_graphs
                        C = graph.input.size(0) // B
                        Z = graph.cond.size(1)
                        latent_dim = graph.cond.size(2)
                        coords = graph.input.view(B, C, -1)
                        features = graph.output.view(B, C, -1)

                        p, _, g = initialise_latents(Z, B, latent_dim, coords.size(-1), cfg.model.spatial_dim, cfg.model.gaussian_window, device, latent_bbox=cfg.model.latent_bbox)
                        p, g = p.to(device), g.to(device)
                        c = graph.cond.to(device)                   
                        features_recon = enf(coords, p, c, g)  # [B, S, out_dim]
                        loss = F.mse_loss(features_recon, features)
                        
                        # In the validation loop, replace your current saving with:
                        all_predictions.append(features_recon.squeeze(0).cpu().numpy())  # Remove batch dimension
                        all_coords.append(coords.squeeze(0).cpu().numpy())              # Remove batch dimension
                        
                            
                    loss_val = loss.cpu().detach()
                    full_field_loss += loss_val.item() * n_samples
                
                test_loss_full_domain = full_field_loss / (nval)
                
                
                test_loss_history.append({'epoch': step, 'loss': test_loss_full_domain})
                wandb.log({"val_loss_Ffull_domain": test_loss_full_domain, "epoch": step})
                print('Full Test Loss', test_loss_in)
            
            # Save predictions for the best model
            if cfg.logging.save_predictions:
                predictions_data = {
                    'predictions': all_predictions,
                    'coordinates': all_coords,
                }
                
                predictions_file = os.path.join(results_directory, f"best_model_predictions.npz")
                np.save(predictions_file, predictions_data)
                print(f"Saved predictions to: {predictions_file}")

            
             
    return


if __name__ == "__main__":
    main()