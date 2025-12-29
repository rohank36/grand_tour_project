from build_dataset import build_offline_dataset, DatasetConfig
from align_data import align_data, load_aligned_data
import numpy as np
import zarr
import h5py
import itertools
from datetime import datetime
from scipy.stats import wasserstein_distance

"""
This script compares the similarity between the grand tour dataset (with different scaling/transformations applied) and the Isaac Gym dataset (target)
"""

def wasserstein_distance_multi(X, Y):
    """
    Compute Wasserstein distance for multi-dimensional arrays.
    Computes 1D Wasserstein distance for each dimension and averages.
    
    Args:
        X: (N, D) array
        Y: (M, D) array
    
    Returns:
        Average Wasserstein distance across dimensions
    """
    min_len = min(len(X), len(Y))
    X = X[:min_len]
    Y = Y[:min_len]
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    # Compute Wasserstein distance for each dimension
    distances = []
    for d in range(X.shape[1]):
        dist = wasserstein_distance(X[:, d], Y[:, d])
        distances.append(dist)
    
    return np.mean(distances)

def mmd_rbf(X, Y, gamma=1.0, max_samples=10000):
    """
    Compute Maximum Mean Discrepancy (MMD) with RBF kernel.
    
    MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    where k(x,y) = exp(-gamma * ||x-y||²)
    
    For large datasets, samples a subset to avoid memory issues.
    
    Args:
        X: (N, D) array
        Y: (M, D) array
        gamma: RBF kernel bandwidth parameter
        max_samples: Maximum number of samples to use (default: 10000)
    
    Returns:
        MMD² value (lower is better = more similar)
    """
    min_len = min(len(X), len(Y))
    X = X[:min_len]
    Y = Y[:min_len]
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    # Sample subset if dataset is too large
    n = len(X)
    if n > max_samples:
        # Randomly sample indices
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(n, size=max_samples, replace=False)
        X = X[indices]
        Y = Y[indices]
        n = max_samples
    
    # RBF kernel: k(x, y) = exp(-gamma * ||x - y||²)
    def rbf_kernel(X1, X2):
        # Compute pairwise squared distances
        # ||x - y||² = ||x||² + ||y||² - 2*x·y
        X1_norm = np.sum(X1**2, axis=1, keepdims=True)  # (N, 1)
        X2_norm = np.sum(X2**2, axis=1, keepdims=True)  # (M, 1)
        X1X2 = X1 @ X2.T  # (N, M)
        sq_dist = X1_norm + X2_norm.T - 2 * X1X2  # (N, M)
        return np.exp(-gamma * sq_dist)
    
    # E[k(x, x')] - average over all pairs in X
    K_XX = rbf_kernel(X, X)
    # Exclude diagonal (self-similarity) for unbiased estimate
    E_XX = (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1)) if n > 1 else 0.0
    
    # E[k(y, y')] - average over all pairs in Y
    K_YY = rbf_kernel(Y, Y)
    E_YY = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1)) if n > 1 else 0.0
    
    # E[k(x, y)] - average over all pairs between X and Y
    K_XY = rbf_kernel(X, Y)
    E_XY = np.mean(K_XY)
    
    # MMD²
    mmd_squared = E_XX - 2 * E_XY + E_YY
    return mmd_squared

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
        cfg = DatasetConfig()  # Create instance with no arguments
        cfg.scale_lin_vel = combo[0]  # Set attributes after creation
        cfg.scale_ang_vel = combo[1]
        cfg.scale_commands = combo[2]
        cfg.scale_joint_pos = combo[3]
        cfg.scale_joint_vel = combo[4]
        cfg.scale_actions = combo[5]
        configs.append(cfg)
    
    return configs

def compare_similarity(aligned_data, ds_cfg, target_obs, target_actions, 
                       obs_mean, obs_std, act_mean, act_std):
    """
    Build dataset with given config and compute similarity metrics.
    Both datasets are normalized using the target dataset statistics.
    
    Args:
        aligned_data: Aligned sensor data
        ds_cfg: Dataset configuration
        target_obs: Target observations (unnormalized)
        target_actions: Target actions (unnormalized)
        obs_mean: Mean for normalizing observations (from target dataset)
        obs_std: Std for normalizing observations (from target dataset)
        act_mean: Mean for normalizing actions (from target dataset)
        act_std: Std for normalizing actions (from target dataset)
    
    Returns:
        obs_wasserstein: Wasserstein distance for observations
        obs_mmd: MMD (RBF) for observations
        act_wasserstein: Wasserstein distance for actions
        act_mmd: MMD (RBF) for actions
        obs_shape: shape of observations
        act_shape: shape of actions
    """
    dataset, _ = build_offline_dataset(aligned_data, ds_cfg)
    obs = dataset["observations"]
    actions = dataset["actions"]
    
    # Normalize both datasets using target statistics
    target_obs_norm = (target_obs - obs_mean) / obs_std
    obs_norm = (obs - obs_mean) / obs_std
    
    target_actions_norm = (target_actions - act_mean) / act_std
    actions_norm = (actions - act_mean) / act_std
    
    # Compute Wasserstein distances on normalized data
    obs_wasserstein = wasserstein_distance_multi(obs_norm, target_obs_norm)
    act_wasserstein = wasserstein_distance_multi(actions_norm, target_actions_norm)
    
    # Compute MMD (RBF) distances on normalized data
    obs_mmd = mmd_rbf(obs_norm, target_obs_norm)
    act_mmd = mmd_rbf(actions_norm, target_actions_norm)
    
    return obs_wasserstein, obs_mmd, act_wasserstein, act_mmd, obs.shape, actions.shape

def load_dataset_lazy(filename):
    """Load dataset from HDF5 file, keeping it open."""
    f = h5py.File(filename, "r")
    return f

if __name__ == "__main__":
    print("="*80)
    print("Dataset Similarity Comparison")
    print("="*80)
    
    # Load data once
    aligned_data_path = "aligned_data.zarr"
    print(f"\nLoading aligned data from {aligned_data_path}...")
    aligned_data = load_aligned_data(aligned_data_path)
    
    target_dataset_path = "expert_dataset.hdf5"
    print(f"Loading target dataset from {target_dataset_path}...")
    target_dataset = load_dataset_lazy(target_dataset_path)
    target_obs = np.array(target_dataset["observations"])
    target_actions = np.array(target_dataset["actions"])
    print(f"Target observations shape: {target_obs.shape}")
    print(f"Target actions shape: {target_actions.shape}")
    
    # Compute normalization statistics from target dataset
    print("Computing normalization statistics from target dataset...")
    obs_mean = np.mean(target_obs, axis=0)
    obs_std = np.std(target_obs, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
    act_mean = np.mean(target_actions, axis=0)
    act_std = np.std(target_actions, axis=0) + 1e-8
    print("Normalization statistics computed.")
    
    # Generate all config combinations
    all_configs = generate_all_configs()
    print(f"\nTesting {len(all_configs)} configuration combinations...")
    print("="*80)
    
    # Results storage: (cfg, obs_wasserstein, obs_mmd, act_wasserstein, act_mmd, obs_shape, act_shape)
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
        f.write(f"Target actions shape: {target_actions.shape}\n")
        f.write(f"Aligned data path: {aligned_data_path}\n")
        f.write(f"Total configurations to test: {len(all_configs)}\n")
        f.write(f"\nNote: All metrics computed on normalized data (normalized using target dataset statistics)\n")
        f.write("="*80 + "\n\n")
        
        # Test each configuration
        for idx, cfg in enumerate(all_configs, 1):
            print(f"[{idx}/{len(all_configs)}] Testing config: {config_to_string(cfg)}")
            
            try:
                obs_w, obs_mmd, act_w, act_mmd, obs_shape, act_shape = compare_similarity(
                    aligned_data, cfg, target_obs, target_actions,
                    obs_mean, obs_std, act_mean, act_std
                )
                results.append((cfg, obs_w, obs_mmd, act_w, act_mmd, obs_shape, act_shape))
                
                # Log to file
                f.write(f"Config {idx:3d}:\n")
                f.write(f"  Obs Wasserstein = {obs_w:.8f}, Obs MMD = {obs_mmd:.8f}\n")
                f.write(f"  Act Wasserstein = {act_w:.8f}, Act MMD = {act_mmd:.8f}\n")
                f.write(f"  Obs shape = {obs_shape}, Act shape = {act_shape}\n")
                f.write(f"  {config_to_string(cfg)}\n")
                f.write("\n")
                
                print(f"  Obs: Wasserstein={obs_w:.8f}, MMD={obs_mmd:.8f}")
                print(f"  Act: Wasserstein={act_w:.8f}, MMD={act_mmd:.8f}")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                f.write(f"Config {idx:3d}: ERROR - {str(e)}\n")
                f.write(f"  {config_to_string(cfg)}\n")
                f.write("\n")
    
    # Sort results by observation similarity (using MMD as primary metric)
    results_obs_sorted = sorted(results, key=lambda x: x[2])  # Sort by obs_mmd
    
    # Sort results by total similarity (obs_mmd + act_mmd)
    results_total_sorted = sorted(results, key=lambda x: x[2] + x[4])  # Sort by obs_mmd + act_mmd
    
    # Write rankings to log file
    with open(log_file, "a") as f:
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 10 RANKED CONFIGURATIONS - OBSERVATION SIMILARITY (Lowest MMD = Most Similar)\n")
        f.write("="*80 + "\n\n")
        
        for rank, (cfg, obs_w, obs_mmd, act_w, act_mmd, obs_shape, act_shape) in enumerate(results_obs_sorted[:10], 1):
            f.write(f"Rank {rank:2d}:\n")
            f.write(f"  Obs Wasserstein = {obs_w:.8f}, Obs MMD = {obs_mmd:.8f}\n")
            f.write(f"  Act Wasserstein = {act_w:.8f}, Act MMD = {act_mmd:.8f}\n")
            f.write(f"  Obs shape = {obs_shape}, Act shape = {act_shape}\n")
            f.write(f"  {config_to_string(cfg)}\n")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 10 RANKED CONFIGURATIONS - TOTAL SIMILARITY (Obs MMD + Act MMD, Lowest = Most Similar)\n")
        f.write("="*80 + "\n\n")
        
        for rank, (cfg, obs_w, obs_mmd, act_w, act_mmd, obs_shape, act_shape) in enumerate(results_total_sorted[:10], 1):
            total_sim = obs_mmd + act_mmd
            f.write(f"Rank {rank:2d} (Total = {total_sim:.8f}):\n")
            f.write(f"  Obs Wasserstein = {obs_w:.8f}, Obs MMD = {obs_mmd:.8f}\n")
            f.write(f"  Act Wasserstein = {act_w:.8f}, Act MMD = {act_mmd:.8f}\n")
            f.write(f"  Obs shape = {obs_shape}, Act shape = {act_shape}\n")
            f.write(f"  {config_to_string(cfg)}\n")
            f.write("\n")
    
    # Print top 10 to console - Observation similarity
    print("\n" + "="*80)
    print("TOP 10 RANKED CONFIGURATIONS - OBSERVATION SIMILARITY (Lowest MMD = Most Similar)")
    print("="*80)
    for rank, (cfg, obs_w, obs_mmd, act_w, act_mmd, obs_shape, act_shape) in enumerate(results_obs_sorted[:10], 1):
        print(f"\nRank {rank:2d}:")
        print(f"  Obs: Wasserstein={obs_w:.8f}, MMD={obs_mmd:.8f}")
        print(f"  Act: Wasserstein={act_w:.8f}, MMD={act_mmd:.8f}")
        print(f"  {config_to_string(cfg)}")
    
    # Print top 10 to console - Total similarity
    print("\n" + "="*80)
    print("TOP 10 RANKED CONFIGURATIONS - TOTAL SIMILARITY (Obs MMD + Act MMD, Lowest = Most Similar)")
    print("="*80)
    for rank, (cfg, obs_w, obs_mmd, act_w, act_mmd, obs_shape, act_shape) in enumerate(results_total_sorted[:10], 1):
        total_sim = obs_mmd + act_mmd
        print(f"\nRank {rank:2d} (Total = {total_sim:.8f}):")
        print(f"  Obs: Wasserstein={obs_w:.8f}, MMD={obs_mmd:.8f}")
        print(f"  Act: Wasserstein={act_w:.8f}, MMD={act_mmd:.8f}")
        print(f"  {config_to_string(cfg)}")
    
    print(f"\n" + "="*80)
    print(f"Results saved to: {log_file}")
    print("="*80)
    
    # Close target dataset
    target_dataset.close()