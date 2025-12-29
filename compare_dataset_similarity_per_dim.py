import numpy as np
from scipy.stats import wasserstein_distance
import h5py

# Observation feature names (36 dimensions)
OBS_FEATURE_NAMES = [
    # 0-2: Base Linear Velocity
    "base_lin_vel_x_body", "base_lin_vel_y_body", "base_lin_vel_z_body",
    # 3-5: Base Angular Velocity
    "base_ang_vel_x_body", "base_ang_vel_y_body", "base_ang_vel_z_body",
    # 6-8: Projected Gravity
    "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
    # 9-11: Commands
    "command_lin_vel_x_body", "command_lin_vel_y_body", "command_ang_vel_yaw_body",
    # 12-23: Joint Positions (12 joints)
    "joint_pos_LF_HAA", "joint_pos_LF_HFE", "joint_pos_LF_KFE",
    "joint_pos_RF_HAA", "joint_pos_RF_HFE", "joint_pos_RF_KFE",
    "joint_pos_LH_HAA", "joint_pos_LH_HFE", "joint_pos_LH_KFE",
    "joint_pos_RH_HAA", "joint_pos_RH_HFE", "joint_pos_RH_KFE",
    # 24-35: Joint Velocities (12 joints)
    "joint_vel_LF_HAA", "joint_vel_LF_HFE", "joint_vel_LF_KFE",
    "joint_vel_RF_HAA", "joint_vel_RF_HFE", "joint_vel_RF_KFE",
    "joint_vel_LH_HAA", "joint_vel_LH_HFE", "joint_vel_LH_KFE",
    "joint_vel_RH_HAA", "joint_vel_RH_HFE", "joint_vel_RH_KFE",
]

# Action feature names (12 dimensions)
# Joint order: ['LF_HAA', 'LF_HFE', 'LF_KFE', 'RF_HAA', 'RF_HFE', 'RF_KFE', 'LH_HAA', 'LH_HFE', 'LH_KFE', 'RH_HAA', 'RH_HFE', 'RH_KFE']
ACTION_FEATURE_NAMES = [
    "action_LF_HAA", "action_LF_HFE", "action_LF_KFE",
    "action_RF_HAA", "action_RF_HFE", "action_RF_KFE",
    "action_LH_HAA", "action_LH_HFE", "action_LH_KFE",
    "action_RH_HAA", "action_RH_HFE", "action_RH_KFE",
]

def wasserstein_per_dimension(X, Y):
    """
    Compute Wasserstein distance for each dimension separately.
    
    Args:
        X: (N, D) array
        Y: (M, D) array
    
    Returns:
        Array of Wasserstein distances, one per dimension
    """
    min_len = min(len(X), len(Y))
    X = X[:min_len]
    Y = Y[:min_len]
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    distances = []
    for d in range(X.shape[1]):
        dist = wasserstein_distance(X[:, d], Y[:, d])
        distances.append(dist)
    
    return np.array(distances)

def mmd_rbf_per_dimension(X, Y, gamma=1.0, max_samples=10000):
    """
    Compute MMD (RBF) for each dimension separately.
    
    Args:
        X: (N, D) array
        Y: (M, D) array
        gamma: RBF kernel bandwidth parameter
        max_samples: Maximum samples for MMD computation
    
    Returns:
        Array of MMD² values, one per dimension
    """
    min_len = min(len(X), len(Y))
    X = X[:min_len]
    Y = Y[:min_len]
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    mmd_values = []
    
    for d in range(X.shape[1]):
        X_dim = X[:, d:d+1]  # Keep as 2D array
        Y_dim = Y[:, d:d+1]
        
        # Sample if needed
        n = len(X_dim)
        if n > max_samples:
            np.random.seed(42)
            indices = np.random.choice(n, size=max_samples, replace=False)
            X_dim = X_dim[indices]
            Y_dim = Y_dim[indices]
            n = max_samples
        
        # RBF kernel for 1D
        def rbf_kernel_1d(X1, X2):
            sq_dist = (X1 - X2.T) ** 2
            return np.exp(-gamma * sq_dist)
        
        # Compute MMD
        K_XX = rbf_kernel_1d(X_dim, X_dim)
        E_XX = (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1)) if n > 1 else 0.0
        
        K_YY = rbf_kernel_1d(Y_dim, Y_dim)
        E_YY = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1)) if n > 1 else 0.0
        
        K_XY = rbf_kernel_1d(X_dim, Y_dim)
        E_XY = np.mean(K_XY)
        
        mmd_squared = E_XX - 2 * E_XY + E_YY
        mmd_values.append(mmd_squared)
    
    return np.array(mmd_values)

def analyze_dimension_shifts(target_data, config_data, data_type="observations"):
    """
    Analyze distribution shifts per dimension.
    
    Args:
        target_data: Target dataset (e.g., target_obs)
        config_data: Config dataset (e.g., config_obs)
        data_type: String for labeling output
    
    Returns:
        Dictionary with per-dimension metrics and rankings
    """
    # Normalize using target statistics
    target_mean = np.mean(target_data, axis=0)
    target_std = np.std(target_data, axis=0) + 1e-8
    
    target_norm = (target_data - target_mean) / target_std
    config_norm = (config_data - target_mean) / target_std
    
    # Compute per-dimension metrics
    wasserstein_per_dim = wasserstein_per_dimension(config_norm, target_norm)
    mmd_per_dim = mmd_rbf_per_dimension(config_norm, target_norm)
    
    # Combined score (weighted: 0.8 Wasserstein, 0.2 MMD)
    # Normalize each metric first
    w_norm = (wasserstein_per_dim - np.min(wasserstein_per_dim)) / (np.max(wasserstein_per_dim) - np.min(wasserstein_per_dim) + 1e-8)
    mmd_norm = (mmd_per_dim - np.min(mmd_per_dim)) / (np.max(mmd_per_dim) - np.min(mmd_per_dim) + 1e-8)
    
    combined_score = w_norm * 0.8 + mmd_norm * 0.2
    
    # Get feature names based on data type
    if data_type.lower() == "observations":
        feature_names = OBS_FEATURE_NAMES
    elif data_type.lower() == "actions":
        feature_names = ACTION_FEATURE_NAMES
    else:
        feature_names = [f"dim_{i}" for i in range(len(wasserstein_per_dim))]
    
    # Ensure we have enough feature names
    if len(feature_names) < len(wasserstein_per_dim):
        feature_names.extend([f"dim_{i}" for i in range(len(feature_names), len(wasserstein_per_dim))])
    
    # Create results dictionary
    results = {
        'wasserstein_per_dim': wasserstein_per_dim,
        'mmd_per_dim': mmd_per_dim,
        'combined_score': combined_score,
        'dimension_rankings': np.argsort(combined_score)[::-1],  # Worst to best
        'feature_names': feature_names,
    }
    
    # Print results
    print(f"\n{'='*80}")
    print(f"{data_type.upper()} - Per-Dimension Distribution Shift Analysis")
    print(f"{'='*80}\n")
    
    print(f"{'Dim':<6} {'Feature Name':<30} {'Wasserstein':<15} {'MMD':<15} {'Combined Score':<15} {'Rank'}")
    print("-" * 100)
    
    for dim_idx in results['dimension_rankings']:
        rank = np.where(results['dimension_rankings'] == dim_idx)[0][0] + 1
        feature_name = feature_names[dim_idx] if dim_idx < len(feature_names) else f"dim_{dim_idx}"
        print(f"{dim_idx:<6} {feature_name:<30} {wasserstein_per_dim[dim_idx]:<15.8f} {mmd_per_dim[dim_idx]:<15.8f} {combined_score[dim_idx]:<15.8f} {rank}")
    
    print(f"\nTop 5 most problematic dimensions (highest shift):")
    for i, dim_idx in enumerate(results['dimension_rankings'][:5], 1):
        feature_name = feature_names[dim_idx] if dim_idx < len(feature_names) else f"dim_{dim_idx}"
        print(f"  {i}. Dim {dim_idx} ({feature_name}): Wasserstein={wasserstein_per_dim[dim_idx]:.6f}, MMD={mmd_per_dim[dim_idx]:.6f}")
    
    return results

def load_dataset_lazy(filename):
    """Load dataset from HDF5 file, keeping it open."""
    f = h5py.File(filename, "r")
    return f

target_dataset_path = "expert_dataset.hdf5"
print(f"Loading target dataset from {target_dataset_path}...")
target_dataset = load_dataset_lazy(target_dataset_path)
target_obs = np.array(target_dataset["observations"])
target_actions = np.array(target_dataset["actions"])
print(f"Target observations shape: {target_obs.shape}")
print(f"Target actions shape: {target_actions.shape}")

gt_dataset_path = "offline_dataset_pp.hdf5"
print(f"Loading grandtour dataset from {gt_dataset_path}...")
gt_dataset = load_dataset_lazy(gt_dataset_path)
gt_obs = np.array(gt_dataset["observations"])
gt_actions = np.array(gt_dataset["actions"])
print(f"Target observations shape: {gt_obs.shape}")
print(f"Target actions shape: {gt_actions.shape}")


results_obs = analyze_dimension_shifts(target_obs, gt_obs, "observations")
results_acts = analyze_dimension_shifts(target_actions, gt_actions, "actions")