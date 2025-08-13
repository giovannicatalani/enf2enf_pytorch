import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import train_test_split
import pyoche as pch
import re
import os
import glob
from collections import defaultdict
import einops


class SDFDataset:
    def __init__(self, data_list, data_names, is_train=True, coef_norm=None, num_points=None):
        """
        Initialize the AirfransDataset class with pre-loaded data.
        
        Args:
            data_list: list of np.arrays containing the raw data
            data_names: list of data names
            target_field: str, one of ['velocity', 'pressure', 'turbulent_viscosity', 'implicit_distance']
            is_train: bool, whether this is training data (to compute normalization)
            coef_norm: dict, normalization coefficients (required if is_train=False)
            num_points: int, number of points to subsample (if None, use all points)
        """
        self.data_list = data_list
        self.data_names = data_names
        self.num_points = num_points
        self.is_train = is_train
        
        
        if is_train:
            # Compute normalization parameters from data
            print("Computing normalization parameters from training data...")
            self.compute_norm_params(data_list)
        else:
            # Use provided normalization parameters
            if coef_norm is None:
                raise ValueError("coef_norm must be provided when is_train=False")
            print("Using provided normalization parameters...")
            self.coef_norm = coef_norm
            self.pos_norm = coef_norm['pos_norm']

        # Process data
        print("Processing dataset...")
        self.processed_dataset = self.process_data(data_list)

    def compute_norm_params(self, data_list):
        """Compute normalization parameters."""
        self.coef_norm = {}
        
        all_targets = np.concatenate([data[:, 4:5] for data in data_list])
        
        self.coef_norm['mean'] = all_targets.mean(axis=0)
        self.coef_norm['std'] = all_targets.std(axis=0)
        
        # Calculate position normalization
        all_positions = np.concatenate([data[:, :2] for data in data_list])
        self.pos_norm = {
            'min': all_positions.min(axis=0),
            'max': all_positions.max(axis=0)
        }
        self.coef_norm['pos_norm'] = self.pos_norm
        
        print("\nNormalization parameters computed:")
        print(f"Target mean: {np.array2string(self.coef_norm['mean'], precision=4, separator=', ')}")
        print(f"Target std: {np.array2string(self.coef_norm['std'], precision=4, separator=', ')}")
        return self.coef_norm

    def process_data(self, dataset_list):
        """Process the raw data into the required format using global normalization."""
        dataset = []
    
        target_idx = 4
        
        # Process each simulation
        for i,sim_data in enumerate(dataset_list):
              
            # Extract and normalize positions
            pos = sim_data[:, :2]
            pos = 2 * (pos - self.pos_norm['min']) / (self.pos_norm['max'] - self.pos_norm['min']) - 1
            input_data = pos
            output = sim_data[:, target_idx:target_idx+1]
            
            # Normalize target using mean and std
            output = (output - self.coef_norm['mean']) / self.coef_norm['std']
            
            # Create data entry
            data_entry = Data(
                pos=torch.tensor(pos, dtype=torch.float),
                input=torch.tensor(input_data, dtype=torch.float),
                output=torch.tensor(output, dtype=torch.float),
                is_airfoil=torch.tensor(sim_data[:, -1]),
            )
            
            dataset.append(data_entry)
        
        # Subsample if requested
        if self.num_points is not None:
            dataset = self.subsample_dataset(dataset, self.num_points)
            
        return dataset

    @staticmethod
    def subsample_dataset(dataset, num_points):
        """Subsample the dataset while maintaining PyG Data format."""
        subsampled_dataset = []
        for data in dataset:
            total_points = data.num_nodes
            
            if total_points > num_points:
                subsample_indices = np.random.choice(total_points, num_points, replace=False)
                
                new_data = Data(
                    input=data.input[subsample_indices],
                    pos=data.pos[subsample_indices],
                    output=data.output[subsample_indices],
                    is_airfoil=data.is_airfoil[subsample_indices],
                )
                
                subsampled_dataset.append(new_data)
            else:
                subsampled_dataset.append(data)
                
        return subsampled_dataset
    
class AirfransFlowDataset(Dataset):
    """
    PyG Dataset for Airfrans flow data.

    Modes:
      - 'train': include inputs, pos, outputs, cond; computes normals
      - 'val':   include inputs, pos, outputs, cond; uses provided coef_norm
      - 'test':  include inputs, pos, cond only; uses provided coef_norm

    Each point in the dataset is described by:
      - Position (2D): indices 0, 1
      - Inlet velocity (2D): indices 2, 3
      - Distance to airfoil (SDF): index 4
      - Normals (2D): indices 5, 6 (set to 0 if point is not on airfoil)
      
    Output fields (target):
      - Velocity (2D): indices 7, 8
      - Pressure/density: index 9
      - Turbulent kinematic viscosity: index 10
      
    Additional:
      - Airfoil boolean: index 11

    Normalizations:
      inputs: [x, y] & sdf -> min-max to [-1, 1]
      outputs: [velocity-x, velocity-y, pressure, turb_viscosity] -> standard
      cond: geom_latents + inlet_velocity -> standard
    """
    def __init__(
        self,
        data_list,
        data_names,
        geom_latents,
        mode='train',
        coef_norm=None,
        num_points=None,
        target_field='all',
        transform=None,
        pre_transform=None
    ):
        super().__init__(None, transform, pre_transform)
        self.data_list = data_list
        self.data_names = data_names
        self.geom_latents = np.array(geom_latents)
        self.mode = mode
        self.num_points = num_points
        self.target_field = target_field
        
        # Define indices
        self.pos_indices = [0, 1]               # Position (x, y)
        self.inlet_vel_indices = [2, 3]         # Inlet velocity components
        self.sdf_index = 4                      # Distance to airfoil
        self.normal_indices = [5, 6]            # Normal components
        self.output_indices = [7, 8, 9, 10]     # Output fields: velocity (2), pressure, viscosity
        self.airfoil_index = 11                 # Airfoil boolean
        
        # Output field names and their corresponding indices
        self.output_fields = ['Velocity-x', 'Velocity-y', 'Pressure', 'Turbulent-viscosity']
        self.field_to_index = {
            'Velocity-x': 0,
            'Velocity-y': 1, 
            'Pressure': 2,
            'Turbulent-viscosity': 3
        }
        
        # Determine which fields to use based on target_field parameter
        if target_field == 'all':
            self.active_fields = self.output_fields.copy()
            self.active_field_indices = list(range(len(self.output_fields)))
        elif isinstance(target_field, str) and target_field in self.field_to_index:
            self.active_fields = [target_field]
            self.active_field_indices = [self.field_to_index[target_field]]
        elif isinstance(target_field, list):
            # Allow multiple specific fields
            self.active_fields = []
            self.active_field_indices = []
            for field in target_field:
                if field in self.field_to_index:
                    self.active_fields.append(field)
                    self.active_field_indices.append(self.field_to_index[field])
                else:
                    raise ValueError(f"Unknown target field: {field}")
        else:
            raise ValueError(f"Invalid target_field: {target_field}. Must be 'all', a field name, or list of field names.")
        
        # norms
        if mode == 'train':
            self.coef_norm = self.compute_norm_params()
        else:
            if coef_norm is None:
                raise ValueError("coef_norm required for non-train modes")
            self.coef_norm = coef_norm
        
        self.data_list = self.process_dataset()

    def compute_norm_params(self):
        """Compute normalization parameters from training data."""
        pos_all = []
        sdf_all = []
        output_fields_all = {i: [] for i in range(len(self.output_fields))}
        inlet_vel_all = []
        cond_all = []
        
        for idx in range(len(self.data_list)):
            data = self.data_list[idx]
            
            # Extract position, SDF, and outputs
            pos = data[:, self.pos_indices]
            sdf = data[:, self.sdf_index]
            outputs = data[:, self.output_indices]
            inlet_vel = data[:, self.inlet_vel_indices]
            
            pos_all.append(pos)
            sdf_all.append(sdf)
            
            if self.mode in ('train', 'val'):
                for i in range(len(self.output_indices)):
                    output_fields_all[i].append(outputs[:, i])
        
            # Collect inlet velocity for conditional input
            avg_inlet_vel = np.mean(inlet_vel, axis=0)  # Shape: [2]
            inlet_vel_all.append(avg_inlet_vel)
            
            # Add latent conditions with inlet velocities for each latent point
            sample_latents = self.geom_latents[idx]  # Shape: [num_latents, latent_dim]
            
            # Concatenate inlet velocity to each latent point
            # Expand inlet_vel to match number of latent points
            inlet_vel_expanded = np.tile(avg_inlet_vel, (sample_latents.shape[0], 1))  # Shape: [num_latents, 2]
            sample_cond = np.concatenate([sample_latents, inlet_vel_expanded], axis=1)  # Shape: [num_latents, latent_dim + 2]
            
            # Flatten to add all latent points to the conditioning data
            cond_all.append(sample_cond.reshape(-1, sample_cond.shape[-1]))  # Shape: [num_latents, latent_dim + 2]
        
        # Process position data
        pos_all = np.vstack(pos_all)
        pos_min = pos_all.min(axis=0)
        pos_max = pos_all.max(axis=0)
        pos_range = pos_max - pos_min
        pos_range[pos_range == 0] = 1.0
        
        # Process SDF data
        sdf_all = np.hstack(sdf_all)
        sdf_min = sdf_all.min()
        sdf_max = sdf_all.max()
        sdf_range = sdf_max - sdf_min
        if sdf_range == 0:
            sdf_range = 1.0
        
        # Process output fields - only compute stats for ALL fields (needed for normalization consistency)
        output_means = {}
        output_stds = {}
        
        if self.mode in ('train', 'val'):
            for i, field_name in enumerate(self.output_fields):
                field_data = np.hstack(output_fields_all[i])
                output_means[field_name] = float(field_data.mean())
                output_stds[field_name] = float(field_data.std())
                if output_stds[field_name] == 0:
                    output_stds[field_name] = 1.0
        
        # Process conditional inputs
        cond_all = np.vstack(cond_all)
        cond_mean = cond_all.mean(axis=0)
        cond_std = cond_all.std(axis=0)
        cond_std[cond_std == 0] = 1.0
        
        # Create and return normalization parameters
        return {
            'pos': {'min': pos_min, 'max': pos_max, 'range': pos_range},
            'sdf': {'min': sdf_min, 'max': sdf_max, 'range': sdf_range},
            'output': {'mean': output_means, 'std': output_stds},
            'cond': {'mean': cond_mean, 'std': cond_std}
        }

    def process_dataset(self):
        """Process the dataset into PyG Data objects."""
        processed_data_list = []
        c = self.coef_norm
        
        for idx in range(len(self.data_list)):
            data = self.data_list[idx]
            
            # Extract required components
            pos = data[:, self.pos_indices]
            sdf = data[:, self.sdf_index]
            airfoil_bool = data[:, self.airfoil_index]
            
            # Subsample if requested
            if self.num_points and pos.shape[0] > self.num_points:
                sel = np.random.choice(pos.shape[0], self.num_points, replace=False)
                pos = pos[sel]
                sdf = sdf[sel]
                airfoil_bool = airfoil_bool[sel]
                # Also subsample outputs if needed
                data = data[sel]
            
            # Normalize position to [-1, 1]
            pmin, pmax = c['pos']['min'], c['pos']['max']
            prange = c['pos']['range']
            pos_n = 2 * (pos - pmin) / prange - 1
            
            # Normalize SDF to [-1, 1]
            smin, srange = c['sdf']['min'], c['sdf']['range']
            sdf_n = 2 * ((sdf - smin) / srange) - 1
            
            # Combine position and SDF as input features
            input_feats = torch.tensor(np.concatenate([pos_n, sdf_n[:, None]], axis=1), dtype=torch.float)
            
            # Prepare node kwargs
            node_kwargs = {
                'input': input_feats,
                'pos': torch.tensor(pos_n, dtype=torch.float),
                'is_airfoil': torch.tensor(airfoil_bool, dtype=torch.float)
            }
            
            
            output_data = []
            for field_name in self.active_fields:
                field_idx = self.field_to_index[field_name]
                field_data = data[:, self.output_indices[field_idx]]
                
                # Normalize the field
                mean_val = c['output']['mean'][field_name]
                std_val = c['output']['std'][field_name]
                field_norm = (field_data - mean_val) / std_val
                output_data.append(field_norm[:, None])
            
            if output_data:
                node_kwargs['output'] = torch.tensor(np.hstack(output_data), dtype=torch.float)
            
            # Add conditional inputs (latent pointcloud + inlet velocity)
            inlet_vel = np.mean(data[:, self.inlet_vel_indices], axis=0)  # Shape: [2]
            sample_latents = self.geom_latents[idx]  # Shape: [num_latents, latent_dim]
            
            # Concatenate inlet velocity to each latent point
            inlet_vel_expanded = np.tile(inlet_vel, (sample_latents.shape[0], 1))  # Shape: [num_latents, 2]
            cond = np.concatenate([sample_latents, inlet_vel_expanded], axis=1)  # Shape: [num_latents, latent_dim + 2]
            
            # Normalize conditional inputs
            cm, cs = c['cond']['mean'], c['cond']['std']
            cond_n = (cond - cm) / cs  # Shape: [num_latents, latent_dim + 2]
            
            node_kwargs['cond'] = torch.tensor(cond_n, dtype=torch.float).unsqueeze(0)  # Shape: [1, num_latents, latent_dim + 2]
            
            processed_data_list.append(Data(**node_kwargs))
        
        return processed_data_list
    
    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]



def subsample_dataset(dataset, num_points, seed=None):
    """
    Subsample each graph in `dataset` to at most `num_points` nodes,
    by randomly selecting rows from `data.input` and `data.output`.
    
    Args:
        dataset: iterable of torch_geometric.data.Data
        num_points: int, max number of nodes per graph
        seed: optional int, for reproducibility
        
    Returns:
        List[Data]: new list of Data objects with subsampled inputs/outputs
    """
    if seed is not None:
        np.random.seed(seed)

    out = []
    for data in dataset:
        N = data.input.size(0)
        if N > num_points:
            # pick random subset of node indices
            idx = np.random.choice(N, num_points, replace=False)
            idx = torch.from_numpy(idx).long()

            # subsample inputs
            inp_sub = data.input[idx]
            pos_sub = data.pos[idx] 
            # subsample outputs if they exist
            out_sub = data.output[idx] if hasattr(data, 'output') else None

            # build new Data
            new_data = Data(input=inp_sub, pos=pos_sub)
            if out_sub is not None:
                new_data.output = out_sub
            # carry through the graph-level fields unchanged
            if hasattr(data, 'cond'):
                new_data.cond = data.cond
            if hasattr(data, 'output_scalars'):
                new_data.output_scalars = data.output_scalars
        else:
            # nothing to do
            new_data = data

        out.append(new_data)
    return out

class ElasticityDataset(Dataset):
    """
    PyG Dataset for Elasticity data.
    
    Modes:
      - 'train': include inputs, pos, outputs, cond; computes normalization
      - 'val':   include inputs, pos, outputs, cond; uses provided coef_norm  
      - 'test':  include inputs, pos, cond only; uses provided coef_norm
    
    Target fields:
      - 'sigma': predicts stress values, uses xy coordinates as input
      - 'xy': predicts deformed coordinates, uses grid as input
    
    When target_field is 'sigma', latents can be provided as conditioning.
    When use_augmented_inputs=True and target_field='sigma', both pos and xy are used as inputs.
    """
    
    def __init__(
        self,
        data_directory=None,
        data_dict=None,  # Alternative to loading from directory
        latents=None,    # Latents for conditioning when target_field='sigma'
        ntrain=1000,
        ntest=200,
        target_field='sigma',
        mode='train',
        coef_norm=None,
        num_points=None,
        use_augmented_inputs=False,  # New parameter for augmented inputs
        transform=None,
        pre_transform=None
    ):
        super().__init__(None, transform, pre_transform)
        
        if target_field not in ['sigma', 'xy']:
            raise ValueError("target_field must be either 'sigma' or 'xy'")
            
        self.data_directory = data_directory
        self.target_field = target_field
        self.ntrain = ntrain
        self.ntest = ntest
        self.mode = mode
        self.num_points = num_points
        self.latents = latents
        self.use_augmented_inputs = use_augmented_inputs
        
        # Load data
        if data_dict is not None:
            self.raw_data = data_dict
        else:
            self.raw_data = self.load_raw_data()
        
        # Compute normalization or use provided
        if mode == 'train':
            self.coef_norm = self.compute_norm_params()
        else:
            if coef_norm is None:
                raise ValueError("coef_norm required for non-train modes")
            self.coef_norm = coef_norm
        
        # Process dataset
        print(f"Processing data for target field: {target_field}, mode: {mode}, augmented inputs: {use_augmented_inputs}...")
        self.data_list = self.process_dataset()

    def load_raw_data(self):
        """Load raw data from files."""
        if self.data_directory is None:
            raise ValueError("Either data_directory or data_dict must be provided")
            
        PATH_Sigma = os.path.join(self.data_directory, "Meshes-20241205T151153Z-001/Meshes/Random_UnitCell_sigma_10.npy")
        PATH_XY = os.path.join(self.data_directory, "Meshes-20241205T151153Z-001/Meshes/Random_UnitCell_XY_10.npy")
        PATH_rr = os.path.join(self.data_directory, "Meshes-20241205T151153Z-001/Meshes/Random_UnitCell_rr_10.npy")
        PATH_theta = os.path.join(self.data_directory, "Meshes-20241205T151153Z-001/Meshes/Random_UnitCell_theta_10.npy")

        return {
            'rr': torch.tensor(np.load(PATH_rr), dtype=torch.float).permute(1, 0),
            'sigma': torch.tensor(np.load(PATH_Sigma), dtype=torch.float).permute(1, 0).unsqueeze(-1),
            'xy': torch.tensor(np.load(PATH_XY), dtype=torch.float).permute(2, 0, 1),
            'theta': torch.tensor(np.load(PATH_theta), dtype=torch.float)
        }

    def compute_norm_params(self):
        """Compute normalization parameters from training data."""
        # Get training split
        train_rr = self.raw_data['rr'][:self.ntrain]
        train_sigma = self.raw_data['sigma'][:self.ntrain]
        train_xy = self.raw_data['xy'][:self.ntrain]
        
        # Get average grid for position normalization
        self.average_grid = train_xy.mean(0)
        
        coef_norm = {}
        
        if self.target_field == 'sigma':
            # Output: Standard normalization (mean, std) for sigma values
            coef_norm['output'] = {
                'mean': float(train_sigma.mean()),
                'std': float(train_sigma.std())
            }
            
            if self.use_augmented_inputs:
                # Input: Concatenated pos and xy coordinates
                # We need to normalize both grid and xy coordinates for input
                grid_flat = self.average_grid.reshape(-1, self.average_grid.shape[-1])
                xy_all = train_xy.reshape(-1, train_xy.shape[-1])
                
                # Combine both for computing joint min/max
                combined_coords = torch.cat([grid_flat.unsqueeze(0).repeat(self.ntrain, 1, 1).reshape(-1, 2), xy_all], dim=0)
                
                coef_norm['input'] = {
                    'min': combined_coords.min(dim=0)[0],
                    'max': combined_coords.max(dim=0)[0]
                }
            else:
                # Input: Min-max normalization to [-1, 1] for xy coordinates only
                xy_all = train_xy.reshape(-1, train_xy.shape[-1])
                coef_norm['input'] = {
                    'min': xy_all.min(dim=0)[0],
                    'max': xy_all.max(dim=0)[0]
                }
            
        else:  # target_field == 'xy'
            # Output: Standard normalization (mean, std) for xy coordinates  
            coef_norm['output'] = {
                'mean': train_xy.mean(dim=(0, 1)),
                'std': train_xy.std(dim=(0, 1))
            }
            
            # Input: Min-max normalization to [-1, 1] for grid coordinates
            grid_flat = self.average_grid.reshape(-1, self.average_grid.shape[-1])
            coef_norm['input'] = {
                'min': grid_flat.min(dim=0)[0],
                'max': grid_flat.max(dim=0)[0]
            }
        
        # Position: Min-max normalization to [-1, 1] (always uses grid)
        grid_flat = self.average_grid.reshape(-1, self.average_grid.shape[-1])
        coef_norm['pos'] = {
            'min': grid_flat.min(dim=0)[0],
            'max': grid_flat.max(dim=0)[0]
        }
        
        # Conditioning: Standard normalization (mean, std) for rr values
        coef_norm['cond'] = {
            'mean': train_rr.mean(dim=0),
            'std': train_rr.std(dim=0)
        }
        
        # If latents are provided for sigma target, use standard normalization
        if self.target_field == 'sigma' and self.latents is not None:
            train_latents = self.latents[:self.ntrain]
            coef_norm['latents'] = {
                'mean': train_latents.mean(dim=(0, 1)),  # Mean over batch and latent points
                'std': train_latents.std(dim=(0, 1))     # Std over batch and latent points
            }
        
        return coef_norm

    def process_dataset(self):
        """Process the dataset into PyG Data objects."""
        processed_data_list = []
        c = self.coef_norm
        
        # Determine data split
        if self.mode == 'train':
            data_slice = slice(0, self.ntrain)
        else:  # val or test
            data_slice = slice(-self.ntest, None)
        
        # Get split data
        split_rr = self.raw_data['rr'][data_slice]
        split_sigma = self.raw_data['sigma'][data_slice]
        split_xy = self.raw_data['xy'][data_slice]
        split_theta = self.raw_data['theta'][data_slice]
        
        # Get average grid (computed during normalization or use existing)
        if hasattr(self, 'average_grid'):
            grid = self.average_grid
        else:
            grid = self.raw_data['xy'][:self.ntrain].mean(0)
        
        batch_size = len(split_rr)
        
        for i in range(batch_size):
            # Prepare position (always uses grid)
            pos = einops.repeat(grid, '... -> b ...', b=1)[0]  # Remove batch dim
            
            if self.target_field == 'sigma':
                if self.use_augmented_inputs:
                    # Input: concatenated pos and xy coordinates
                    xy_coords = split_xy[i]
                    # Concatenate along feature dimension: [N_points, 4] where first 2 are pos, last 2 are xy
                    input_data = torch.cat([pos, xy_coords], dim=-1)
                else:
                    # Input: xy coordinates only
                    input_data = split_xy[i]
                # Output: sigma values
                output_data = split_sigma[i]
            else:  # target_field == 'xy'
                # Input: grid coordinates, Output: xy coordinates
                input_data = pos.clone()
                output_data = split_xy[i]
            
            # Subsample if requested
            if self.num_points and input_data.shape[0] > self.num_points:
                sel = np.random.choice(input_data.shape[0], self.num_points, replace=False)
                pos = pos[sel]
                input_data = input_data[sel]
                if self.mode in ('train', 'val'):
                    output_data = output_data[sel]
            
            # Normalize position (min-max to [-1, 1])
            pos_range = c['pos']['max'] - c['pos']['min']
            pos_range[pos_range == 0] = 1.0  # Avoid division by zero
            pos_norm = 2 * (pos - c['pos']['min']) / pos_range - 1
            
            # Normalize input
            if self.target_field == 'sigma' and self.use_augmented_inputs:
                # For augmented inputs, normalize each coordinate pair separately but with same stats
                input_range = c['input']['max'] - c['input']['min']
                input_range[input_range == 0] = 1.0
                
                # Normalize pos part (first 2 dimensions)
                input_pos_norm = 2 * (input_data[..., :2] - c['input']['min']) / input_range - 1
                # Normalize xy part (last 2 dimensions) 
                input_xy_norm = 2 * (input_data[..., 2:] - c['input']['min']) / input_range - 1
                # Concatenate normalized parts
                input_norm = torch.cat([input_pos_norm, input_xy_norm], dim=-1)
            else:
                # Standard normalization for non-augmented inputs
                input_range = c['input']['max'] - c['input']['min']
                input_range[input_range == 0] = 1.0
                input_norm = 2 * (input_data - c['input']['min']) / input_range - 1
            
            # Prepare node kwargs
            node_kwargs = {
                'pos': pos_norm,
                'input': input_norm,
            }
            
            # Add output for train/val modes
            if self.mode in ('train', 'val'):
                output_norm = (output_data - c['output']['mean']) / c['output']['std']
                node_kwargs['output'] = output_norm
            
            # Prepare conditioning
            cond_data = split_rr[i]
            cond_norm = (cond_data - c['cond']['mean']) / c['cond']['std']
            
            # If latents are provided for sigma target, combine with rr
            if self.target_field == 'sigma' and self.latents is not None:
                sample_latents = self.latents[data_slice][i]  # Get latents for this sample
                latents_norm = (sample_latents - c['latents']['mean']) / c['latents']['std']
                
                # Combine latents with rr conditioning
                # Option 1: Concatenate rr to each latent point
                node_kwargs['cond'] = latents_norm.unsqueeze(0)  # Add batch dim
            else:
                # Just use rr as conditioning
                node_kwargs['cond'] = cond_norm.unsqueeze(0).unsqueeze(0)  # Add batch and seq dims
            
            # Add theta information (might be useful for some applications)
            node_kwargs['theta'] = split_theta[i].unsqueeze(0)  # Add batch dim
            
            processed_data_list.append(Data(**node_kwargs))
        
        return processed_data_list
    
    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
    
    def get_normalization_stats(self):
        """Get normalization statistics for compatibility."""
        return self.coef_norm
    

class MultiElementSDFDataset:
    def __init__(self, database_path, is_train=True, coef_norm=None, num_points=None, 
                 train_split=0.8, random_state=42):
        """
        Efficient SDF Dataset that only loads unique geometry samples.
        
        Args:
            database_path: str, path to the pyoche database folder
            is_train: bool, whether this is training data
            coef_norm: dict, normalization coefficients (required if is_train=False)
            num_points: int, number of points to subsample (if None, use all points)
            train_split: float, fraction of data to use for training (default 0.8)
            random_state: int, random seed for reproducible splits
        """
        self.database_path = database_path
        self.num_points = num_points
        self.is_train = is_train
        
        # Find all sample files without loading them
        print(f"Scanning for sample files in {database_path}...")
        self.all_sample_files = self.find_sample_files()
        print(f"Found {len(self.all_sample_files)} total sample files")
        
        # Group files by unique geometry and select representatives
        self.select_unique_geometries()
        
        # Split unique geometries into train/test
        self.split_geometries(train_split, random_state)
        
        # Select appropriate split
        if is_train:
            self.sample_files = self.train_files
            print(f"Using {len(self.train_files)} unique training geometries")
        else:
            self.sample_files = self.test_files
            print(f"Using {len(self.test_files)} unique test geometries")
        
        # Compute or use normalization parameters
        if is_train:
            print("Computing normalization parameters from training geometries...")
            self.compute_norm_params()
        else:
            if coef_norm is None:
                raise ValueError("coef_norm must be provided when is_train=False")
            print("Using provided normalization parameters...")
            self.coef_norm = coef_norm
            self.pos_norm = coef_norm['pos_norm']

        # Load and process only the required samples
        print("Loading and processing selected samples...")
        self.processed_dataset = self.load_and_process_samples()
        
    def find_sample_files(self):
        """Find all .h5 sample files without loading them."""
        pattern = os.path.join(self.database_path, "**/*.h5")
        return glob.glob(pattern, recursive=True)
        
    def extract_geometry_key(self, filepath):
        """Extract geometry key from filepath."""
        filename = os.path.basename(filepath)
        
        # Extract config type (LDG, CLEAN, TO)
        config_match = re.search(r'sample_(LDG|CLEAN|TO)_', filename)
        config_type = config_match.group(1) if config_match else None
        
        # Extract y location
        y_match = re.search(r'y([\d.]+)', filename)
        if y_match:
            y_str = y_match.group(1).rstrip('.')
            try:
                y_location = float(y_str)
            except ValueError:
                y_location = None
        else:
            y_location = None
        
        if config_type and y_location is not None:
            return f"{config_type}_y{y_location:.3f}"
        else:
            return None
    
    def extract_alpha(self, filepath):
        """Extract alpha value from filepath."""
        filename = os.path.basename(filepath)
        alpha_match = re.search(r'alpha([-]?[\d.]+)', filename)
        if alpha_match:
            alpha_str = alpha_match.group(1).rstrip('.')
            try:
                return float(alpha_str)
            except ValueError:
                return None
        return None
    
    def select_unique_geometries(self):
        """Select one representative file per unique geometry."""
        geometry_groups = defaultdict(list)
        
        # Group files by geometry
        for filepath in self.all_sample_files:
            geometry_key = self.extract_geometry_key(filepath)
            if geometry_key:
                alpha = self.extract_alpha(filepath)
                geometry_groups[geometry_key].append((filepath, alpha))
        
        # Select representative file for each geometry (prefer alpha=0)
        self.unique_geometry_files = {}
        self.geometry_stats = {}
        
        for geometry_key, files_with_alpha in geometry_groups.items():
            # Try to find alpha=0 first
            representative = None
            best_alpha_diff = float('inf')
            
            for filepath, alpha in files_with_alpha:
                if alpha is not None:
                    alpha_diff = abs(alpha)
                    if alpha_diff < best_alpha_diff:
                        best_alpha_diff = alpha_diff
                        representative = filepath
            
            # If no valid alpha found, use first file
            if representative is None and files_with_alpha:
                representative = files_with_alpha[0][0]
            
            if representative:
                self.unique_geometry_files[geometry_key] = representative
                self.geometry_stats[geometry_key] = {
                    'total_files': len(files_with_alpha),
                    'representative': os.path.basename(representative),
                    'representative_alpha': self.extract_alpha(representative)
                }
        
        print(f"Selected {len(self.unique_geometry_files)} unique geometries from {len(self.all_sample_files)} files")
        print(f"Reduction factor: {len(self.all_sample_files) / len(self.unique_geometry_files):.1f}x")
        
        # Print geometry distribution
        config_counts = defaultdict(int)
        for geometry_key in self.unique_geometry_files.keys():
            config = geometry_key.split('_')[0]
            config_counts[config] += 1
        
        print("Geometry distribution:")
        for config, count in config_counts.items():
            print(f"  {config}: {count} unique geometries")
    
    def split_geometries(self, train_split, random_state):
        """Split unique geometries into train and test sets."""
        geometry_keys = list(self.unique_geometry_files.keys())
        
        # Split geometry keys
        train_keys, test_keys = train_test_split(
            geometry_keys, 
            test_size=1-train_split, 
            random_state=random_state
        )
        
        # Get corresponding files
        self.train_files = [self.unique_geometry_files[key] for key in train_keys]
        self.test_files = [self.unique_geometry_files[key] for key in test_keys]
        
        print(f"Geometry split: {len(self.train_files)} train, {len(self.test_files)} test")
    
    def compute_norm_params(self):
        """Compute normalization parameters from training files only."""
        print("Loading training samples for normalization...")
        
        all_sdf = []
        all_positions = []
        
        for i, filepath in enumerate(self.train_files):
            print(f"  Loading {i+1}/{len(self.train_files)}: {os.path.basename(filepath)}")
            try:
                sample = pch.Sample.from_file(filepath)
                all_sdf.append(sample['point_cloud_field/sdf'])
                all_positions.append(sample['point_cloud_field/coordinates'])
            except Exception as e:
                print(f"    Error loading {filepath}: {e}")
                continue
        
        if not all_sdf:
            raise ValueError("No valid training samples found for normalization")
        
        all_sdf = np.concatenate(all_sdf)
        all_positions = np.concatenate(all_positions)
        
        self.coef_norm = {
            'mean': all_sdf.mean(),
            'std': all_sdf.std()
        }
        
        self.pos_norm = {
            'min': all_positions.min(axis=0),
            'max': all_positions.max(axis=0)
        }
        self.coef_norm['pos_norm'] = self.pos_norm
        
        print(f"SDF normalization - mean: {self.coef_norm['mean']:.4f}, std: {self.coef_norm['std']:.4f}")
        print(f"Position range: [{self.pos_norm['min'][0]:.3f}, {self.pos_norm['min'][1]:.3f}] to [{self.pos_norm['max'][0]:.3f}, {self.pos_norm['max'][1]:.3f}]")
        
        return self.coef_norm
    
    def load_and_process_samples(self):
        """Load and process only the selected sample files."""
        dataset = []
        
        for i, filepath in enumerate(self.sample_files):
            print(f"Processing {i+1}/{len(self.sample_files)}: {os.path.basename(filepath)}")
            
            try:
                # Load sample
                sample = pch.Sample.from_file(filepath)
                
                # Extract data
                points = sample['point_cloud_field/coordinates']  # [N, 2]
                sdf = sample['point_cloud_field/sdf']            # [N, 1] or [N]
                
                # Flatten SDF if needed
                if sdf.ndim > 1:
                    sdf = sdf.flatten()
                
                # Normalize positions to [-1, 1]
                pos = 2 * (points - self.pos_norm['min']) / (self.pos_norm['max'] - self.pos_norm['min']) - 1
                
                # Normalize SDF
                normalized_sdf = (sdf - self.coef_norm['mean']) / self.coef_norm['std']
                
                # Create is_airfoil mask (1 for inside geometry, 0 for outside)
                is_airfoil = (sdf < 0).astype(float)
                
                # Create PyTorch Geometric Data object
                data_entry = Data(
                    pos=torch.tensor(pos, dtype=torch.float),
                    input=torch.tensor(pos, dtype=torch.float),
                    output=torch.tensor(normalized_sdf.reshape(-1, 1), dtype=torch.float),
                    is_airfoil=torch.tensor(is_airfoil, dtype=torch.float),
                    filename=os.path.basename(filepath),
                    geometry_key=self.extract_geometry_key(filepath),
                )
                
                dataset.append(data_entry)
                
            except Exception as e:
                print(f"  Error processing {filepath}: {e}")
                continue
        
        # Subsample if requested
        if self.num_points is not None:
            print(f"Subsampling to {self.num_points} points per sample...")
            dataset = self.subsample_dataset(dataset, self.num_points)
        
        print(f"Successfully processed {len(dataset)} samples")
        return dataset
    
    @staticmethod
    def subsample_dataset(dataset, num_points):
        """Subsample the dataset while maintaining PyG Data format."""
        subsampled_dataset = []
        for data in dataset:
            total_points = data.num_nodes
            
            if total_points > num_points:
                subsample_indices = np.random.choice(total_points, num_points, replace=False)
                
                new_data = Data(
                    input=data.input[subsample_indices],
                    pos=data.pos[subsample_indices],
                    output=data.output[subsample_indices],
                    is_airfoil=data.is_airfoil[subsample_indices],
                    filename=data.filename,
                    geometry_key=data.geometry_key,
                )
                
                subsampled_dataset.append(new_data)
            else:
                subsampled_dataset.append(data)
                
        return subsampled_dataset
    
    def get_normalization_stats(self):
        """Return normalization statistics."""
        return self.coef_norm
    
    def get_geometry_info(self):
        """Return information about selected geometries."""
        return self.geometry_stats
    
    def __len__(self):
        return len(self.processed_dataset)
    
    def __getitem__(self, idx):
        return self.processed_dataset[idx]

class MultiElementFlowDataset(Dataset):
    """
    PyG Dataset for Multi-Element Airfoil Flow data.
    """
    
    def __init__(
        self,
        database,  # Pre-loaded database instead of path
        geom_latents,
        mode='train',
        coef_norm=None,
        num_points=None,
        target_field='PressureCoefficient',
        alpha_range=None,  # Tuple (min_alpha, max_alpha) to filter alpha values
        config_types=['CLEAN'],  # List of config types to include
        transform=None,
        pre_transform=None
    ):
        super().__init__(None, transform, pre_transform)
        
        self.database = database  # Accept pre-loaded database
        self.geom_latents = geom_latents
        self.mode = mode
        self.num_points = num_points
        self.target_field = target_field
        self.alpha_range = alpha_range
        self.config_types = config_types  # Make config filtering configurable
        
        # Extract geometry keys from latents
        self.geometry_keys = geom_latents['geometry_keys']
        self.latent_embeddings = geom_latents['latents'].detach().numpy()
        
        print(f"Found {len(self.geometry_keys)} geometries in latents")
        print(f"Loaded {len(self.database)} samples from database")
        print(f"Filtering for config types: {self.config_types}")
        if alpha_range:
            print(f"Filtering for alpha range: [{alpha_range[0]:.1f}, {alpha_range[1]:.1f}]")
        
        # Filter samples
        print("Filtering samples...")
        self.filtered_sample_indices, self.sample_to_geom_idx = self.filter_samples()
        print(f"Found {len(self.filtered_sample_indices)} matching samples")
        
        # Compute normalization or use provided
        if mode == 'train':
            self.coef_norm = self.compute_norm_params()
        else:
            if coef_norm is None:
                raise ValueError("coef_norm required for non-train modes")
            self.coef_norm = coef_norm
        
        # Process dataset
        print(f"Processing data for mode: {mode}...")
        self.data_list = self.process_dataset()
        print(f"Successfully processed {len(self.data_list)} samples")

    def extract_geometry_key(self, sample_name):
        """Extract geometry key from sample name."""
        # Extract config type and check if it's in our allowed list
        config_match = re.search(r'sample_(LDG|CLEAN|TO)_', sample_name)
        if not config_match:
            return None
            
        config_type = config_match.group(1)
        
        # Filter by allowed config types
        if config_type not in self.config_types:
            return None
        
        # Extract y location
        y_match = re.search(r'y([\d.]+)', sample_name)
        if y_match:
            y_str = y_match.group(1).rstrip('.')
            try:
                y_location = float(y_str)
            except ValueError:
                return None
        else:
            return None
        
        if config_type and y_location is not None:
            return f"{config_type}_y{y_location:.3f}"
        else:
            return None
    
    def extract_alpha(self, sample_name):
        """Extract alpha value from sample name."""
        alpha_match = re.search(r'alpha([-]?[\d.]+)', sample_name)
        if alpha_match:
            alpha_str = alpha_match.group(1).rstrip('.')
            try:
                return float(alpha_str)
            except ValueError:
                return None
        return None

    def filter_samples(self):
        """Filter samples based on geometry keys, config types, and alpha range."""
        # Create mapping from geometry key to index in latents
        geom_key_to_idx = {key[0]: idx for idx, key in enumerate(self.geometry_keys)}
        
        filtered_sample_indices = []
        sample_to_geom_idx = []
        filtered_by_alpha = 0
        filtered_by_geom = 0
        filtered_by_config = 0
        
        for i, sample in enumerate(self.database):
            sample_name = sample.name
            
            # Check config type first (this handles the CLEAN filtering)
            config_match = re.search(r'sample_(LDG|CLEAN|TO)_', sample_name)
            if not config_match or config_match.group(1) not in self.config_types:
                filtered_by_config += 1
                continue
            
            # Check geometry
            geom_key = self.extract_geometry_key(sample_name)
            if not geom_key or geom_key not in geom_key_to_idx:
                filtered_by_geom += 1
                continue
            
            # Check alpha range if specified
            if self.alpha_range is not None:
                alpha = self.extract_alpha(sample_name)
                if alpha is None or alpha < self.alpha_range[0] or alpha > self.alpha_range[1]:
                    filtered_by_alpha += 1
                    continue
            
            # Sample passes all filters
            filtered_sample_indices.append(i)
            sample_to_geom_idx.append(geom_key_to_idx[geom_key])
        
        print(f"Filtering results:")
        print(f"  Total samples in database: {len(self.database)}")
        print(f"  Filtered by config type: {filtered_by_config}")
        print(f"  Filtered by geometry: {filtered_by_geom}")
        if self.alpha_range:
            print(f"  Filtered by alpha range: {filtered_by_alpha}")
        print(f"  Final matching samples: {len(filtered_sample_indices)}")
        
        # Print sample distribution
        geom_file_counts = defaultdict(int)
        alpha_values = defaultdict(list)
        
        for sample_idx, geom_idx in zip(filtered_sample_indices, sample_to_geom_idx):
            sample_name = self.database[sample_idx].name
            geom_key = self.geometry_keys[geom_idx][0]
            alpha = self.extract_alpha(sample_name)
            geom_file_counts[geom_key] += 1
            if alpha is not None:
                alpha_values[geom_key].append(alpha)
        
        print(f"Sample distribution (after filtering):")
        for geom_key in self.geometry_keys[:5]:  # Show first 5
            count = geom_file_counts[geom_key[0]]
            alphas = sorted(alpha_values[geom_key[0]]) if geom_key[0] in alpha_values else []
            alpha_range = f"α∈[{min(alphas):.1f}, {max(alphas):.1f}]" if alphas else "no α"
            print(f"  {geom_key[0]}: {count} samples, {alpha_range}")
        
        if len(self.geometry_keys) > 5:
            print(f"  ... and {len(self.geometry_keys) - 5} more geometries")
        
        return filtered_sample_indices, sample_to_geom_idx

    def compute_norm_params(self):
        """Compute normalization parameters from training data."""
        print("Computing normalization parameters...")
        
        pos_all = []
        sdf_all = []
        output_all = []
        alpha_all = []
        cond_all = []
        
        # Sample a subset of samples for normalization to speed up computation
        sample_size = min(1000, len(self.filtered_sample_indices))
        sample_indices = np.random.choice(len(self.filtered_sample_indices), sample_size, replace=False)
        
        for i in sample_indices:
            sample_idx = self.filtered_sample_indices[i]
            geom_idx = self.sample_to_geom_idx[i]
            
            try:
                sample = self.database[sample_idx]
                sample_name = sample.name
                
                # Extract data
                points = sample['point_cloud_field/coordinates']
                sdf_data = sample['point_cloud_field/sdf'] 
                output_data = sample[f'point_cloud_field/{self.target_field}']
                alpha = self.extract_alpha(sample_name)
                
                if output_data.ndim > 1:
                    output_data = output_data.flatten()
                if sdf_data.ndim > 1:
                    sdf_data = sdf_data.flatten()
                
                pos_all.append(points)
                sdf_all.append(sdf_data)
                output_all.append(output_data)
                
                if alpha is not None:
                    alpha_all.append(alpha)
                
                # Get corresponding latents
                sample_latents = self.latent_embeddings[geom_idx]
                alpha_val = alpha if alpha is not None else 0.0
                
                # Create conditioning: concatenate alpha to each latent point
                alpha_expanded = np.full((sample_latents.shape[0], 1), alpha_val)
                sample_cond = np.concatenate([sample_latents, alpha_expanded], axis=1)
                cond_all.append(sample_cond)
                
            except Exception as e:
                print(f"  Warning: Error loading sample {sample_idx}: {e}")
                continue
        
        if not pos_all:
            raise ValueError("No valid training samples found for normalization")
        
        # Process position data
        pos_all = np.vstack(pos_all)
        pos_min = pos_all.min(axis=0)
        pos_max = pos_all.max(axis=0)
        pos_range = pos_max - pos_min
        pos_range[pos_range == 0] = 1.0
        
        # Process SDF data
        sdf_all = np.hstack(sdf_all)
        sdf_min = float(sdf_all.min())
        sdf_max = float(sdf_all.max())
        sdf_range = sdf_max - sdf_min
        if sdf_range == 0:
            sdf_range = 1.0
        
        # Process output data
        output_all = np.hstack(output_all)
        output_mean = float(output_all.mean())
        output_std = float(output_all.std())
        if output_std == 0:
            output_std = 1.0
        
        # Process conditioning data
        cond_all = np.vstack(cond_all)
        cond_mean = cond_all.mean(axis=0)
        cond_std = cond_all.std(axis=0)
        cond_std[cond_std == 0] = 1.0
        
        # Alpha statistics
        if alpha_all:
            alpha_mean = float(np.mean(alpha_all))
            alpha_std = float(np.std(alpha_all))
            if alpha_std == 0:
                alpha_std = 1.0
        else:
            alpha_mean, alpha_std = 0.0, 1.0
        
        coef_norm = {
            'pos': {'min': pos_min, 'max': pos_max, 'range': pos_range},
            'sdf': {'min': sdf_min, 'max': sdf_max, 'range': sdf_range},
            'output': {'mean': output_mean, 'std': output_std},
            'cond': {'mean': cond_mean, 'std': cond_std},
            'alpha': {'mean': alpha_mean, 'std': alpha_std}
        }
        
        print(f"Normalization computed from {len(pos_all)} points across {sample_size} samples:")
        print(f"  Output: mean={output_mean:.4f}, std={output_std:.4f}")
        print(f"  SDF range: [{sdf_min:.3f}, {sdf_max:.3f}]")
        print(f"  Alpha: mean={alpha_mean:.2f}, std={alpha_std:.2f}")
        print(f"  Position range: [{pos_min[0]:.3f}, {pos_min[1]:.3f}] to [{pos_max[0]:.3f}, {pos_max[1]:.3f}]")
        
        return coef_norm

    def process_dataset(self):
        """Process the dataset into PyG Data objects."""
        processed_data_list = []
        c = self.coef_norm
        
        for i, sample_idx in enumerate(self.filtered_sample_indices):
            if i % 50 == 0:
                print(f"  Processing {i+1}/{len(self.filtered_sample_indices)}")
            
            geom_idx = self.sample_to_geom_idx[i]
            
            try:
                # Load sample
                sample = self.database[sample_idx]
                sample_name = sample.name
                
                # Extract data
                points = sample['point_cloud_field/coordinates']
                sdf_data = sample['point_cloud_field/sdf']
                alpha = self.extract_alpha(sample_name)
                alpha_val = alpha if alpha is not None else 0.0
                
                if sdf_data.ndim > 1:
                    sdf_data = sdf_data.flatten()
                
                # Subsample if requested
                if self.num_points and points.shape[0] > self.num_points:
                    sel = np.random.choice(points.shape[0], self.num_points, replace=False)
                    points = points[sel]
                    sdf_data = sdf_data[sel]
                
                # Normalize position to [-1, 1]
                pos_norm = 2 * (points - c['pos']['min']) / c['pos']['range'] - 1
                
                # Normalize SDF to [-1, 1]
                sdf_norm = 2 * ((sdf_data - c['sdf']['min']) / c['sdf']['range']) - 1
                
                # Combine position and SDF as input features [N_points, 3] -> [x, y, sdf]
                input_features = np.concatenate([pos_norm, sdf_norm[:, None]], axis=1)
                
                # Prepare node kwargs
                node_kwargs = {
                    'pos': torch.tensor(pos_norm, dtype=torch.float),
                    'input': torch.tensor(input_features, dtype=torch.float),  # [x, y, sdf]
                    'alpha': torch.tensor([alpha_val], dtype=torch.float),
                    'geometry_key': self.geometry_keys[geom_idx],
                    'sample_name': sample_name
                }
                
                # Add output for train/val modes
                if self.mode in ('train', 'val'):
                    output_data = sample[f'point_cloud_field/{self.target_field}']
                    if output_data.ndim > 1:
                        output_data = output_data.flatten()
                    
                    if self.num_points and len(output_data) > self.num_points:
                        output_data = output_data[sel]
                    
                    # Normalize output
                    output_norm = (output_data - c['output']['mean']) / c['output']['std']
                    node_kwargs['output'] = torch.tensor(output_norm.reshape(-1, 1), dtype=torch.float)
                
                # Add conditioning (geometry latents + alpha)
                sample_latents = self.latent_embeddings[geom_idx]  # [N_latent_points, latent_dim]
                
                # Concatenate alpha to each latent point
                alpha_expanded = np.full((sample_latents.shape[0], 1), alpha_val)
                cond = np.concatenate([sample_latents, alpha_expanded], axis=1)
                
                # Normalize conditioning
                cond_norm = (cond - c['cond']['mean']) / c['cond']['std']
                node_kwargs['cond'] = torch.tensor(cond_norm, dtype=torch.float).unsqueeze(0)  # Add batch dim
                
                processed_data_list.append(Data(**node_kwargs))
                
            except Exception as e:
                print(f"  Warning: Error processing sample {sample_idx}: {e}")
                continue
        
        return processed_data_list
    
    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
    
    def get_normalization_stats(self):
        """Get normalization statistics for compatibility."""
        return self.coef_norm

class RotorFlowDataset(Dataset):
    """
    PyG Dataset for Rotor Flow data.
    
    Modes:
      - 'train': include inputs, pos, outputs, cond; computes normalization
      - 'val':   include inputs, pos, outputs, cond; uses provided coef_norm  
      - 'test':  include inputs, pos, cond only; uses provided coef_norm
    
    Each point in the dataset is described by:
      - Position (3D coordinates): used as pos
      - Input features: Position + Normals (6D total)
      - Output fields: Density, Pressure, Temperature
      - Conditioning: geometry latents + operating conditions (Omega, P)
      - Scalar outputs: Massflow, Compression_ratio, Efficiency
    """
    
    def __init__(
        self,
        ml_dataset,
        idxs,
        geom_latents,
        mode='train',
        coef_norm=None,
        num_points=None,
        transform=None,
        pre_transform=None
    ):
        super().__init__(None, transform, pre_transform)
        
        self.ml_dataset = ml_dataset
        self.idxs = idxs
        self.geom_latents = geom_latents
        self.mode = mode
        self.num_points = num_points
        
        # Extract latent embeddings from the geom_latents dict
        self.latent_embeddings = geom_latents['latents'].detach().numpy()
        
        # Scalar field keys
        self.scalar_keys = ['Massflow', 'Compression_ratio', 'Efficiency']
        
        print(f"Loaded latents for {len(self.idxs)} rotor samples")
        print(f"Latent embeddings shape: {self.latent_embeddings.shape}")
        
        # Compute normalization or use provided
        if mode == 'train':
            self.coef_norm = self.compute_norm_params()
        else:
            if coef_norm is None:
                raise ValueError("coef_norm required for non-train modes")
            self.coef_norm = coef_norm
        
        # Process dataset
        print(f"Processing rotor data for mode: {mode}...")
        self.data_list = self.process_dataset()
        print(f"Successfully processed {len(self.data_list)} samples")

    def compute_norm_params(self):
        """Compute normalization parameters from training data."""
        print("Computing normalization parameters for rotor data...")
        
        pos_all = []
        normals_all = []
        rho_all, pressure_all, temperature_all = [], [], []
        cond_all = []
        scalar_all = []
        
        for i, idx in enumerate(self.idxs):
            item = self.ml_dataset[idx]
            
            # Extract position
            pts = np.array(item.get_nodes())  # Shape: [N_points, 3]
            pos_all.append(pts)
            
            # Extract normals
            normals_x = np.array(item.get_field('NormalsX')).T.flatten()
            normals_y = np.array(item.get_field('NormalsY')).T.flatten()
            normals_z = np.array(item.get_field('NormalsZ')).T.flatten()
            normals = np.stack([normals_x, normals_y, normals_z], axis=1)
            normals_all.append(normals)
            
            # Extract output fields for train/val modes
            if self.mode in ('train', 'val'):
                rho = np.array(item.get_field('Density')).T.flatten()
                pressure = np.array(item.get_field('Pressure')).T.flatten()
                temperature = np.array(item.get_field('Temperature')).T.flatten()
                
                rho_all.append(rho)
                pressure_all.append(pressure)
                temperature_all.append(temperature)
                
                # Extract scalar outputs
                scalars = np.array([float(item.get_scalar(k)) for k in self.scalar_keys])
                scalar_all.append(scalars)
            
            # Extract operating conditions
            omega = float(item.get_scalar('Omega'))
            p_val = float(item.get_scalar('P'))
            operating_conditions = np.array([omega, p_val])
            
            # Get geometry latents for this sample
            sample_latents = self.latent_embeddings[i]  # Shape: [n_latent_points, latent_dim]
            
            # Concatenate operating conditions to each latent point
            op_cond_expanded = np.tile(operating_conditions, (sample_latents.shape[0], 1))
            sample_cond = np.concatenate([sample_latents, op_cond_expanded], axis=1)
            
            cond_all.append(sample_cond)
        
        # Process position data
        pos_all = np.vstack(pos_all)
        pos_min = pos_all.min(axis=0)
        pos_max = pos_all.max(axis=0)
        pos_range = pos_max - pos_min
        pos_range[pos_range == 0] = 1.0
        
        # Process normals data
        normals_all = np.vstack(normals_all)
        normals_min = normals_all.min(axis=0)
        normals_max = normals_all.max(axis=0)
        normals_range = normals_max - normals_min
        normals_range[normals_range == 0] = 1.0
        
        # Process output fields
        output_means = {}
        output_stds = {}
        
        if self.mode in ('train', 'val'):
            # Field outputs (per-point)
            rho_all = np.hstack(rho_all)
            pressure_all = np.hstack(pressure_all)
            temperature_all = np.hstack(temperature_all)
            
            output_means['field'] = np.array([rho_all.mean(), pressure_all.mean(), temperature_all.mean()])
            output_stds['field'] = np.array([rho_all.std(), pressure_all.std(), temperature_all.std()])
            output_stds['field'][output_stds['field'] == 0] = 1.0
            
            # Scalar outputs (per-sample)
            scalar_all = np.vstack(scalar_all)
            output_means['scalar'] = scalar_all.mean(axis=0)
            output_stds['scalar'] = scalar_all.std(axis=0)
            output_stds['scalar'][output_stds['scalar'] == 0] = 1.0
        
        # Process conditioning data
        cond_all = np.vstack(cond_all)
        cond_mean = cond_all.mean(axis=0)
        cond_std = cond_all.std(axis=0)
        cond_std[cond_std == 0] = 1.0
        
        coef_norm = {
            'pos': {'min': pos_min, 'max': pos_max, 'range': pos_range},
            'normals': {'min': normals_min, 'max': normals_max, 'range': normals_range},
            'output': {'mean': output_means, 'std': output_stds},
            'cond': {'mean': cond_mean, 'std': cond_std}
        }
        
        print(f"Normalization computed from {len(pos_all)} points across {len(self.idxs)} samples:")
        print(f"  Position range: [{pos_min[0]:.3f}, {pos_min[1]:.3f}, {pos_min[2]:.3f}] to [{pos_max[0]:.3f}, {pos_max[1]:.3f}, {pos_max[2]:.3f}]")
        print(f"  Normals range: [{normals_min[0]:.3f}, {normals_min[1]:.3f}, {normals_min[2]:.3f}] to [{normals_max[0]:.3f}, {normals_max[1]:.3f}, {normals_max[2]:.3f}]")
        if self.mode in ('train', 'val'):
            print(f"  Field outputs - Density: mean={output_means['field'][0]:.4f}, std={output_stds['field'][0]:.4f}")
            print(f"  Field outputs - Pressure: mean={output_means['field'][1]:.4f}, std={output_stds['field'][1]:.4f}")
            print(f"  Field outputs - Temperature: mean={output_means['field'][2]:.4f}, std={output_stds['field'][2]:.4f}")
            print(f"  Scalar outputs - Massflow: mean={output_means['scalar'][0]:.4f}, std={output_stds['scalar'][0]:.4f}")
        
        return coef_norm

    def process_dataset(self):
        """Process the dataset into PyG Data objects."""
        processed_data_list = []
        c = self.coef_norm
        
        for i, idx in enumerate(self.idxs):
            if i % 50 == 0 and len(self.idxs) > 50:
                print(f"  Processing {i+1}/{len(self.idxs)}")
            
            try:
                item = self.ml_dataset[idx]
                
                # Extract position
                pts = np.array(item.get_nodes())  # Shape: [N_points, 3]
                
                # Extract normals
                normals_x = np.array(item.get_field('NormalsX')).T.flatten()
                normals_y = np.array(item.get_field('NormalsY')).T.flatten()
                normals_z = np.array(item.get_field('NormalsZ')).T.flatten()
                normals = np.stack([normals_x, normals_y, normals_z], axis=1)
                    
                # Subsample if requested
                if self.num_points and pts.shape[0] > self.num_points:
                    sel = np.random.choice(pts.shape[0], self.num_points, replace=False)
                    pts = pts[sel]
                    normals = normals[sel]
                
                # Normalize position to [-1, 1]
                pos_norm = 2 * (pts - c['pos']['min']) / c['pos']['range'] - 1
                
                # Normalize normals to [-1, 1]
                normals_norm = 2 * (normals - c['normals']['min']) / c['normals']['range'] - 1
                
                # Combine position and normals as input features [N_points, 6]
                input_feats = np.concatenate([pos_norm, normals_norm], axis=1)
                
                # Prepare node kwargs
                node_kwargs = {
                    'pos': torch.tensor(pos_norm, dtype=torch.float),
                    'input': torch.tensor(input_feats, dtype=torch.float),
                }
                
                # Add field outputs for train/val modes
                if self.mode in ('train', 'val'):
                    # Extract field data
                    rho = np.array(item.get_field('Density')).T.flatten()
                    pressure = np.array(item.get_field('Pressure')).T.flatten()
                    temperature = np.array(item.get_field('Temperature')).T.flatten()
                    
                    # Subsample outputs if points were subsampled
                    if self.num_points and len(rho) > self.num_points:
                        rho = rho[sel]
                        pressure = pressure[sel]
                        temperature = temperature[sel]
                    
                    # Normalize field outputs
                    field_means = c['output']['mean']['field']
                    field_stds = c['output']['std']['field']
                    
                    rho_norm = (rho - field_means[0]) / field_stds[0]
                    pressure_norm = (pressure - field_means[1]) / field_stds[1]
                    temperature_norm = (temperature - field_means[2]) / field_stds[2]
                    
                    # Stack field outputs [N_points, 3]
                    field_outputs = np.stack([rho_norm, pressure_norm, temperature_norm], axis=1)
                    node_kwargs['output'] = torch.tensor(field_outputs, dtype=torch.float)
                    
                    # Add scalar outputs
                    scalars = np.array([float(item.get_scalar(k)) for k in self.scalar_keys])
                    scalar_means = c['output']['mean']['scalar']
                    scalar_stds = c['output']['std']['scalar']
                    scalars_norm = (scalars - scalar_means) / scalar_stds
                    node_kwargs['output_scalars'] = torch.tensor(scalars_norm, dtype=torch.float).unsqueeze(0)
                
                # Add conditioning (geometry latents + operating conditions)
                omega = float(item.get_scalar('Omega'))
                p_val = float(item.get_scalar('P'))
                operating_conditions = np.array([omega, p_val])
                
                # Get geometry latents for this sample
                sample_latents = self.latent_embeddings[i]  # Shape: [n_latent_points, latent_dim]
                
                # Concatenate operating conditions to each latent point
                op_cond_expanded = np.tile(operating_conditions, (sample_latents.shape[0], 1))
                cond = np.concatenate([sample_latents, op_cond_expanded], axis=1)
                
                # Normalize conditioning
                cond_norm = (cond - c['cond']['mean']) / c['cond']['std']
                node_kwargs['cond'] = torch.tensor(cond_norm, dtype=torch.float).unsqueeze(0)
                
                processed_data_list.append(Data(**node_kwargs))
                
            except Exception as e:
                print(f"  Warning: Error processing sample {idx}: {e}")
                continue
        
        return processed_data_list
    
    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
    
    def get_normalization_stats(self):
        """Get normalization statistics for compatibility."""
        return self.coef_norm
    
    
class RotorSDFDataset(Dataset):
    """
    PyG Dataset for 2D PROFILE data loaded via pyoche MlDataset.

    Each sample in ml_dataset should support indexing and provide:
      - item['point_cloud_field/coordinates']: numpy array of shape (M, 3?) or (M, 2) for coordinates
      - item['point_cloud_field/sdf']: numpy array of shape (M,) of SDF values

    Args:
        ml_dataset: pyoche.MlDataset instance (or similar) with ['mesh/points'], ['mesh/sdf']
        is_train: whether to compute normalization from this dataset
        coef_norm: precomputed normalization dict (required if is_train=False)
        num_points: subsample each sample to this number of points
        transform, pre_transform: for PyG compatibility
    """
    def __init__(self, ml_dataset, is_train=True, coef_norm=None, num_points=None,
                 transform=None, pre_transform=None):
        super(RotorSDFDataset, self).__init__(None, transform, pre_transform)
        self.ml_dataset = ml_dataset
        self.num_points = num_points
        self.is_train = is_train

        if self.is_train:
            self.coef_norm = self.compute_norm_params()
        else:
            if coef_norm is None:
                raise ValueError("coef_norm must be provided when is_train is False.")
            self.coef_norm = coef_norm

        self.data_list = self.process_dataset()

    def compute_norm_params(self):
        """Compute normalization parameters (pos min/max, sdf mean/std) from all samples."""
        all_positions = []
        all_sdf = []
        for idx in range(len(self.ml_dataset)):
            item = self.ml_dataset[idx]
            pts = np.array(item['point_cloud_field/coordinates'])
            sdf = np.array(item['point_cloud_field/sdf']).T
            all_positions.append(pts)
            all_sdf.append(sdf)
        all_positions = np.vstack(all_positions)
        all_sdf = np.vstack(all_sdf)

        pos_min = all_positions.min(axis=0)
        pos_max = all_positions.max(axis=0)
        sdf_mean = all_sdf.mean(axis=0)
        sdf_std = all_sdf.std(axis=0)

        # Avoid zero division
        pos_range = pos_max - pos_min
        pos_range[pos_range == 0] = 1.0
        if sdf_std == 0:
            sdf_std = 1.0

        coef_norm = {
            'pos_norm': {
                'min': pos_min,
                'max': pos_max
            },
            'mean': sdf_mean,
            'std': sdf_std
        }

        print("Normalization parameters computed:")
        print(f"Position min: {pos_min}, max: {pos_max}")
        print(f"SDF mean: {sdf_mean}, std: {sdf_std}")
        return coef_norm

    def process_dataset(self):
        """Normalize, filter, subsample and convert each sample to a PyG Data object."""
        data_list = []
        pos_min = self.coef_norm['pos_norm']['min']
        pos_max = self.coef_norm['pos_norm']['max']
        sdf_mean = self.coef_norm['mean']
        sdf_std = self.coef_norm['std']

        # Handle zero-range safeguards
        pos_range = pos_max - pos_min
        pos_range[pos_range == 0] = 1.0

        for idx in range(len(self.ml_dataset)):
            item = self.ml_dataset[idx]
            pts = np.array(item['point_cloud_field/coordinates'])
            sdf = np.array(item['point_cloud_field/sdf']).T

            # Normalize positions to [-1, 1]
            pts_norm = 2 * (pts - pos_min) / pos_range - 1
            # Normalize SDF to zero mean, unit std
            sdf_norm = (sdf - sdf_mean) / sdf_std

        
            # Subsample if requested
            if self.num_points is not None and pts_norm.shape[0] > self.num_points:
                indices = np.random.choice(pts_norm.shape[0], self.num_points, replace=False)
                pts_norm = pts_norm[indices]
                sdf_norm = sdf_norm[indices]

            data = Data(
                pos=torch.tensor(pts_norm, dtype=torch.float),
                input=torch.tensor(pts_norm, dtype=torch.float),
                output=torch.tensor(sdf_norm, dtype=torch.float),
                is_airfoil=torch.zeros((pts_norm.shape[0],), dtype=torch.float)
            )
            data_list.append(data)
        return data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# Example usage
def main():
    """Example of how to use the efficient dataset."""
    
    # # Create training dataset (loads only unique geometries)
    # train_dataset = MultiElementSDFDataset(
    #     database_path="/scratch/dmsm/gi.catalani/NASA_CRM_HL/pyoche_dataset/airfoil_dataset.pch",
    #     is_train=True,
    #     num_points=None,
    #     train_split=0.8
    # )
    
    # # Get normalization parameters
    # norm_params = train_dataset.get_normalization_stats()
    
    # # Create test dataset
    # test_dataset = MultiElementSDFDataset(
    #     database_path="/scratch/dmsm/gi.catalani/NASA_CRM_HL/pyoche_dataset/airfoil_dataset.pch",
    #     is_train=False,
    #     coef_norm=norm_params,
    #     num_points=None,
    #     train_split=0.8
    # )
    
    # print(f"\nFinal dataset:")
    # print(f"Training samples: {len(train_dataset)}")
    # print(f"Test samples: {len(test_dataset)}")
    
    # # Show first sample
    # if len(train_dataset) > 0:
    #     sample = train_dataset[0]
    #     print(f"\nFirst training sample:")
    #     print(f"  Geometry: {sample.geometry_key}")
    #     print(f"  Filename: {sample.filename}")
    #     print(f"  Points: {sample.num_nodes}")
    #     print(f"  SDF range: {sample.output.min():.3f} to {sample.output.max():.3f}")
    
    
    # Example usage:
    
    database = pch.MlDataset.from_folder("/scratch/dmsm/gi.catalani/NASA_CRM_HL/pyoche_dataset/airfoil_dataset.pch", load_subset=1000)
    

    
    train_latents = torch.load('/home/dmsm/gi.catalani/Projects/enf-torch/enf-torch-main/multi_airfoil/experiments/training/2025-07-18_16-23-42/modulations/train_modulations.pt')
    val_latents = torch.load('/home/dmsm/gi.catalani/Projects/enf-torch/enf-torch-main/multi_airfoil/experiments/training/2025-07-18_16-23-42/modulations/val_modulations.pt')
    
    train_dataset_out = MultiElementFlowDataset(database=database,config_types=['CLEAN'],
                                                geom_latents=train_latents,alpha_range=[-5,15], mode='train')
    # Get normalization parameters
    norm_params = train_dataset_out.get_normalization_stats()
    val_dataset_out = MultiElementFlowDataset(database=database,config_types=['CLEAN'],
                                                geom_latents=val_latents,coef_norm=norm_params, alpha_range=[-5,15],mode='test')
    
    
    print(f"\nFinal dataset:")
    print(f"Training samples: {len(train_dataset_out)}")
    print(f"Test samples: {len(val_dataset_out)}")
    print(f"Out shape: {train_dataset_out[0].output.shape}")
    print(f"Cond shape: {train_dataset_out[0].cond.shape}")
    
    


if __name__ == "__main__":
    main()