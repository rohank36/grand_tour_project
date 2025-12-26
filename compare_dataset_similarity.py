from build_dataset import build_offline_dataset, DatasetConfig
from align_data import align_data, load_aligned_data
import numpy as np
import zarr
import h5py
import itertools
from datetime import datetime

"""
This script compares the similarity between the grand tour dataset (with different scaling/transformations applied) and the Isaac Gym dataset (target)
"""

def mse_distance(A, B):
    """Compute mean squared error between two arrays."""
    # Handle different lengths by taking the minimum
    min_len = min(len(A), len(B))
    return np.mean((A[:min_len] - B[:min_len]) ** 2)

def config_to_string(cfg):
    """Convert DatasetConfig to a readable string representation."""
    return (f"scale_lin_vel={cfg.scale_lin_vel}, "
            f"scale_ang_vel={cfg.scale_ang_vel}, "
            f"scale_commands={cfg.scale_commands}, "
            f"scale_joint_pos={cfg.scale_joint_pos}, "
            f"scale_joint_vel={cfg.scale_joint_vel}, "
            f"scale_actions={cfg.scale_actions}")

def generate_all_configs():
    """Generate all 2^6 = 64 combinations of boolean scaling flags."""
    flags = ['scale_lin_vel', 'scale_ang_vel', 'scale_commands', 
             'scale_joint_pos', 'scale_joint_vel', 'scale_actions']
    
    configs = []
    # Generate all combinations: 2^6 = 64
    for combo in itertools.product([False, True], repeat=6):
        cfg = DatasetConfig(
            scale_lin_vel=combo[0],
            scale_ang_vel=combo[1],
            scale_commands=combo[2],
            scale_joint_pos=combo[3],
            scale_joint_vel=combo[4],
            scale_actions=combo[5]
        )
        configs.append(cfg)
    
    return configs

def compare_obs_similarity(aligned_data, ds_cfg, target_obs):
    """Build dataset with given config and compute MSE similarity to target observations."""
    dataset, _ = build_offline_dataset(aligned_data, ds_cfg)
    obs = dataset["observations"]
    
    # Compute MSE distance (lower is better = more similar)
    mse = mse_distance(obs, target_obs)
    
    return mse, obs.shape

def load_dataset_lazy(filename):
    """Load dataset from HDF5 file, keeping it open."""
    f = h5py.File(filename, "r")
    return f

if __name__ == "__main__":
    print("="*80)
    print("Dataset Similarity Comparison")
    print("="*80)
    
    # Load data once
    aligned_data_path = "aligned_dataset.zarr"
    print(f"\nLoading aligned data from {aligned_data_path}...")
    aligned_data = load_aligned_data(aligned_data_path)
    
    target_dataset_path = "expert_dataset.hdf5"
    print(f"Loading target dataset from {target_dataset_path}...")
    target_dataset = load_dataset_lazy(target_dataset_path)
    target_obs = np.array(target_dataset["observations"])
    print(f"Target observations shape: {target_obs.shape}")
    
    # Generate all config combinations
    all_configs = generate_all_configs()
    print(f"\nTesting {len(all_configs)} configuration combinations...")
    print("="*80)
    
    # Results storage
    results = []
    log_file = "dataset_similarity_results.txt"
    
    # Open log file for writing
    with open(log_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("Dataset Similarity Comparison Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Target dataset: {target_dataset_path}\n")
        f.write(f"Target observations shape: {target_obs.shape}\n")
        f.write(f"Aligned data path: {aligned_data_path}\n")
        f.write(f"Total configurations to test: {len(all_configs)}\n")
        f.write("="*80 + "\n\n")
        
        # Test each configuration
        for idx, cfg in enumerate(all_configs, 1):
            print(f"[{idx}/{len(all_configs)}] Testing config: {config_to_string(cfg)}")
            
            try:
                mse, obs_shape = compare_obs_similarity(aligned_data, cfg, target_obs)
                results.append((cfg, mse, obs_shape))
                
                # Log to file
                f.write(f"Config {idx:3d}: MSE = {mse:.8f}, Obs shape = {obs_shape}\n")
                f.write(f"  {config_to_string(cfg)}\n")
                f.write("\n")
                
                print(f"  MSE: {mse:.8f}, Obs shape: {obs_shape}")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                f.write(f"Config {idx:3d}: ERROR - {str(e)}\n")
                f.write(f"  {config_to_string(cfg)}\n")
                f.write("\n")
    
    # Sort results by MSE (lower is better = more similar)
    results.sort(key=lambda x: x[1])
    
    # Write top 10 results to log file
    with open(log_file, "a") as f:
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 10 RANKED CONFIGURATIONS (Lowest MSE = Most Similar)\n")
        f.write("="*80 + "\n\n")
        
        for rank, (cfg, mse, obs_shape) in enumerate(results[:10], 1):
            f.write(f"Rank {rank:2d}: MSE = {mse:.8f}, Obs shape = {obs_shape}\n")
            f.write(f"  {config_to_string(cfg)}\n")
            f.write("\n")
    
    # Print top 10 to console
    print("\n" + "="*80)
    print("TOP 10 RANKED CONFIGURATIONS (Lowest MSE = Most Similar)")
    print("="*80)
    for rank, (cfg, mse, obs_shape) in enumerate(results[:10], 1):
        print(f"\nRank {rank:2d}: MSE = {mse:.8f}, Obs shape = {obs_shape}")
        print(f"  {config_to_string(cfg)}")
    
    print(f"\n" + "="*80)
    print(f"Results saved to: {log_file}")
    print("="*80)
    
    # Close target dataset
    target_dataset.close()