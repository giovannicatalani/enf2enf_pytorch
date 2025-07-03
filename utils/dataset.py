from torch_geometric.data import Data, Dataset
import torch
import numpy as np
import einops
import os

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
    
    
from sklearn.model_selection import train_test_split

class MultiElementSDFDataset:
    def __init__(self, database_path, is_train=True, coef_norm=None, num_points=None, 
                 train_split=0.8, random_state=42):
        """
        Initialize the MultiElementSDFDataset from saved database.
        
        Args:
            database_path: str, path to the saved .npy database file
            is_train: bool, whether this is training data
            coef_norm: dict, normalization coefficients (required if is_train=False)
            num_points: int, number of points to subsample (if None, use all points)
            train_split: float, fraction of data to use for training (default 0.8)
            random_state: int, random seed for reproducible splits
        """
        self.num_points = num_points
        self.is_train = is_train
        
        # Load database
        print(f"Loading database from {database_path}...")
        self.database = np.load(database_path, allow_pickle=True).item()
        print(f"Loaded {len(self.database)} geometries")
        
        # Print config distribution
        config_counts = {}
        for entry in self.database.values():
            config_type = entry.get('config_type', 'UNKNOWN')
            config_counts[config_type] = config_counts.get(config_type, 0) + 1
        print("Configuration distribution:")
        for config_type, count in sorted(config_counts.items()):
            print(f"  {config_type}: {count} geometries")
        
        # Split data into train/test
        self.split_data(train_split, random_state)
        
        # Select appropriate split
        if is_train:
            self.data_list = self.train_data
            print(f"Using {len(self.train_data)} training samples")
        else:
            self.data_list = self.test_data
            print(f"Using {len(self.test_data)} test samples")
        
        # Compute or use normalization parameters
        if is_train:
            print("Computing normalization parameters from training data...")
            self.compute_norm_params()
        else:
            if coef_norm is None:
                raise ValueError("coef_norm must be provided when is_train=False")
            print("Using provided normalization parameters...")
            self.coef_norm = coef_norm
            self.pos_norm = coef_norm['pos_norm']

        # Process data
        print("Processing dataset...")
        self.processed_dataset = self.process_data()
        
    def split_data(self, train_split, random_state):
        """Split the database into train and test sets."""
        indices = list(self.database.keys())
        
        # Split indices
        train_indices, test_indices = train_test_split(
            indices, 
            test_size=1-train_split, 
            random_state=random_state
        )
        
        # Extract training and test data
        self.train_data = [self.database[idx] for idx in train_indices]
        self.test_data = [self.database[idx] for idx in test_indices]
        
        print(f"Data split: {len(self.train_data)} train, {len(self.test_data)} test")
        
    def compute_norm_params(self):
        """Compute normalization parameters from training data."""
        self.coef_norm = {}
        
        # Concatenate all SDF values
        all_sdf = np.concatenate([entry['sdf'] for entry in self.train_data])
        
        self.coef_norm['mean'] = all_sdf.mean()
        self.coef_norm['std'] = all_sdf.std()
        
        # Calculate position normalization
        all_positions = np.concatenate([entry['points'] for entry in self.train_data])
        self.pos_norm = {
            'min': all_positions.min(axis=0),
            'max': all_positions.max(axis=0)
        }
        self.coef_norm['pos_norm'] = self.pos_norm
        
        print(f"SDF mean: {self.coef_norm['mean']:.4f}, std: {self.coef_norm['std']:.4f}")
        print(f"Position range: [{self.pos_norm['min'][0]:.3f}, {self.pos_norm['min'][1]:.3f}] to [{self.pos_norm['max'][0]:.3f}, {self.pos_norm['max'][1]:.3f}]")
        
        return self.coef_norm

    def process_data(self):
        """Process the data into PyTorch Geometric format."""
        dataset = []
        
        for entry in self.data_list:
            # Extract data from database entry
            points = entry['points']  # [N, 2] - x, y coordinates
            sdf = entry['sdf']       # [N] - SDF values
            
            # Normalize positions to [-1, 1]
            pos = 2 * (points - self.pos_norm['min']) / (self.pos_norm['max'] - self.pos_norm['min']) - 1
            
            # Normalize SDF
            normalized_sdf = (sdf - self.coef_norm['mean']) / self.coef_norm['std']
            
            # Create is_airfoil mask (1 for inside geometry, 0 for outside)
            is_airfoil = (sdf < 0).astype(float)
            
            # Create PyTorch Geometric Data object
            data_entry = Data(
                pos=torch.tensor(pos, dtype=torch.float),
                input=torch.tensor(pos, dtype=torch.float),  # Use positions as input features
                output=torch.tensor(normalized_sdf.reshape(-1, 1), dtype=torch.float),
                is_airfoil=torch.tensor(is_airfoil, dtype=torch.float),
                filename=entry['filename'],
                config_type=entry.get('config_type', 'UNKNOWN')
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
                    filename=data.filename,
                    config_type=data.config_type
                )
                
                subsampled_dataset.append(new_data)
            else:
                subsampled_dataset.append(data)
                
        return subsampled_dataset
    
    def get_normalization_stats(self):
        """Return normalization statistics (for compatibility)."""
        return self.coef_norm
    
    def __len__(self):
        return len(self.processed_dataset)
    
    def __getitem__(self, idx):
        return self.processed_dataset[idx]


if __name__ == "__main__":
    
    dataset_name = "multi_element_sdf"
    
    if dataset_name == "multi_element_sdf":
        # Testing without the helper function
        database_path = "/home/dmsm/gi.catalani/Projects/enf-torch/enf-torch-main/multi_airfoil/data/multi_element_sdf_database.npy"
        
        print("=== Testing MultiElementSDFDataset ===")
        
        # Create training dataset first
        print("\n1. Creating training dataset...")
        train_dataset = MultiElementSDFDataset(
            database_path=database_path,
            is_train=True,
            num_points=None,  # Use all points
            train_split=0.8,
            random_state=42
        )
        
        print(f"Training dataset created with {len(train_dataset)} samples")
        
        # Create test dataset using training normalization
        print("\n2. Creating test dataset...")
        test_dataset = MultiElementSDFDataset(
            database_path=database_path,
            is_train=False,
            coef_norm=train_dataset.coef_norm,  # Use training normalization
            num_points=None,
            train_split=0.8,
            random_state=42
        )
        
    elif dataset_name == "elasticity":
        
        train_dataset = ElasticityDataset(
            data_directory='/scratch/dmsm/gi.catalani/Elasticity/',
            target_field='xy',
            mode='train',
            ntrain=1000,
            ntest=200
            )

        # Get normalization for val/test
        norm_stats = train_dataset.get_normalization_stats()

        test_dataset = ElasticityDataset(
            data_directory='/scratch/dmsm/gi.catalani/Elasticity/',
            target_field='xy',
            mode='val',
            coef_norm=norm_stats,
            ntrain=1000,
            ntest=200
            )
        
    
    print(f"Test dataset created with {len(test_dataset)} samples")
    
    # Examine a training sample
    print("\n3. Examining training sample...")
    train_sample = train_dataset[0]
    print(f"Sample filename: {train_sample.filename}")
    print(f"Sample config_type: {train_sample.config_type}")
    print(f"Input shape: {train_sample.input.shape}")
    print(f"Output shape: {train_sample.output.shape}")
    print(f"Position shape: {train_sample.pos.shape}")
    print(f"Is_airfoil shape: {train_sample.is_airfoil.shape}")
    print(f"Number of points inside airfoil: {train_sample.is_airfoil.sum().item()}")
   
    
    # Check normalization ranges
    print("\n4. Checking normalization...")
    print(f"Position range: [{train_sample.pos.min():.3f}, {train_sample.pos.max():.3f}]")
    print(f"SDF output range: [{train_sample.output.min():.3f}, {train_sample.output.max():.3f}]")
    
    # Test PyTorch Geometric DataLoader
    print("\n5. Testing with PyTorch Geometric DataLoader...")
    from torch_geometric.loader import DataLoader
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Get first batch
    batch = next(iter(train_loader))
    print(f"Batch input shape: {batch.input.shape}")
    print(f"Batch output shape: {batch.output.shape}")
    print(f"Batch size: {batch.num_graphs}")
    print(f"Total points in batch: {batch.num_nodes}")
    
    # Test accessing different samples
    print("\n6. Testing sample access...")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        print(f"Sample {i}: {sample.filename}, {sample.config_type}, {sample.input.shape[0]} points")
    
    print("\n7. Normalization coefficients:")
    print(f"SDF mean: {train_dataset.coef_norm['mean']:.6f}")
    print(f"SDF std: {train_dataset.coef_norm['std']:.6f}")
    print(f"Position normalization: {train_dataset.coef_norm['pos_norm']}")
    
    print("\n=== Testing completed successfully! ===")