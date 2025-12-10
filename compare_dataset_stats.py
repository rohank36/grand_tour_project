import numpy as np
import h5py
import os
import matplotlib.pyplot as plt  # <-- use pyplot


def load_dataset_lazy(filename):
    f = h5py.File(filename, "r")   # keep file open
    return f   # dict-like interface


def episode_returns(rewards, terminals):
    episode_sums = []
    current_sum = 0.0

    for r, done in zip(rewards, terminals):
        current_sum += r
        if done:
            episode_sums.append(current_sum)
            current_sum = 0.0

    if not terminals[-1]:
        episode_sums.append(current_sum)

    return np.array(episode_sums)


def get_obs_stats(dataset):
    obs = dataset["observations"]

    stats = {
        'mean': np.mean(obs, axis=0),
        'std': np.std(obs, axis=0),
        'min': np.min(obs, axis=0),
        'max': np.max(obs, axis=0)
    }

    return stats


def get_act_stats(dataset):
    actions = dataset["actions"]

    stats = {
        'mean': np.mean(actions, axis=0),
        'std': np.std(actions, axis=0),
        'min': np.min(actions, axis=0),
        'max': np.max(actions, axis=0)
    }

    return stats


def print_obs_stats(dataset1, dataset2, name1="Isaac Gym", name2="Grand Tour"):
    """Print formatted observation statistics comparing two datasets"""
    stats1 = get_obs_stats(dataset1)
    stats2 = get_obs_stats(dataset2)

    obs_dim = len(stats1['mean'])
    assert obs_dim == len(stats2['mean']), "Datasets must have same observation dimension"

    print("="*120)
    print("Observation Statistics Comparison (per dimension):")
    print("="*120)
    print(f"{'Dim':<6} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15} | "
          f"{'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print(f"{'':6} {name1:<60} | {name2:<60}")
    print("-"*120)

    for i in range(obs_dim):
        print(f"{i:<6} {stats1['mean'][i]:<15.6f} {stats1['std'][i]:<15.6f} "
              f"{stats1['min'][i]:<15.6f} {stats1['max'][i]:<15.6f} | "
              f"{stats2['mean'][i]:<15.6f} {stats2['std'][i]:<15.6f} "
              f"{stats2['min'][i]:<15.6f} {stats2['max'][i]:<15.6f}")

    print("="*120)
    print("Overall Statistics:")
    print(f"  {name1}:")
    print(f"    Mean: {stats1['mean'].mean():.6f}, Std: {stats1['std'].mean():.6f}, "
          f"Min: {stats1['min'].min():.6f}, Max: {stats1['max'].max():.6f}")
    print(f"  {name2}:")
    print(f"    Mean: {stats2['mean'].mean():.6f}, Std: {stats2['std'].mean():.6f}, "
          f"Min: {stats2['min'].min():.6f}, Max: {stats2['max'].max():.6f}")
    print("="*120)


def print_act_stats(dataset1, dataset2, name1="Isaac Gym", name2="Grand Tour"):
    """Print formatted action statistics comparing two datasets"""
    stats1 = get_act_stats(dataset1)
    stats2 = get_act_stats(dataset2)

    act_dim = len(stats1['mean'])
    assert act_dim == len(stats2['mean']), "Datasets must have same action dimension"

    print("="*120)
    print("Action Statistics Comparison (per dimension):")
    print("="*120)
    print(f"{'Dim':<6} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15} | "
          f"{'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print(f"{'':6} {name1:<60} | {name2:<60}")
    print("-"*120)

    for i in range(act_dim):
        print(f"{i:<6} {stats1['mean'][i]:<15.6f} {stats1['std'][i]:<15.6f} "
              f"{stats1['min'][i]:<15.6f} {stats1['max'][i]:<15.6f} | "
              f"{stats2['mean'][i]:<15.6f} {stats2['std'][i]:<15.6f} "
              f"{stats2['min'][i]:<15.6f} {stats2['max'][i]:<15.6f}")

    print("="*120)
    print("Overall Statistics:")
    print(f"  {name1}:")
    print(f"    Mean: {stats1['mean'].mean():.6f}, Std: {stats1['std'].mean():.6f}, "
          f"Min: {stats1['min'].min():.6f}, Max: {stats1['max'].max():.6f}")
    print(f"  {name2}:")
    print(f"    Mean: {stats2['mean'].mean():.6f}, Std: {stats2['std'].mean():.6f}, "
          f"Min: {stats2['min'].min():.6f}, Max: {stats2['max'].max():.6f}")
    print("="*120)


def plot_acts(dir_path, ds1, ds2, n1="Isaac Gym", n2="Grand Tour", max_points=100_000):
    """
    - Take up to `max_points` first rows from each dataset.
    - For each action dimension i, make a line plot:
        y1 = ds1.actions[:N, i]
        y2 = ds2.actions[:N, i]
    - Save each plot under dir_path as 'actions_dim_i.png'.
    """
    os.makedirs(dir_path, exist_ok=True)

    a1 = ds1["actions"]   # h5py dataset, not loaded yet
    a2 = ds2["actions"]

    N1, D1 = a1.shape
    N2, D2 = a2.shape
    assert D1 == D2, "Action dimensions must match between datasets"

    act_dim = D1
    N = min(N1, N2, max_points)

    # x-axis: just 0..N-1
    x = np.arange(N)

    print(f"Plotting first {N} samples for {act_dim} action dimensions...")

    for i in range(act_dim):
        # read only the needed slice from disk (fast and memory-light)
        y1 = a1[:N, i]
        y2 = a2[:N, i]

        plt.figure()
        plt.plot(x, y1, label=n1)
        plt.plot(x, y2, label=n2)
        plt.xlabel("timestep (index)")
        plt.ylabel(f"action dim {i}")
        plt.title(f"Actions comparison - dim {i}")
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(dir_path, f"actions_dim_{i}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Plot {i} done.")

def plot_obs(dir_path, ds1, ds2, n1="Isaac Gym", n2="Grand Tour", max_points=100_000):
   
    os.makedirs(dir_path, exist_ok=True)

    o1 = ds1["observations"]   # h5py dataset, not loaded yet
    o2 = ds2["observations"]

    N1, D1 = o1.shape
    N2, D2 = o2.shape
    assert D1 == D2, "Action dimensions must match between datasets"

    obs_dim = D1
    N = min(N1, N2, max_points)

    # x-axis: just 0..N-1
    x = np.arange(N)

    print(f"Plotting first {N} samples for {obs_dim} obs dimensions...")

    for i in range(obs_dim):
        # read only the needed slice from disk (fast and memory-light)
        y1 = o1[:N, i]
        y2 = o2[:N, i]

        plt.figure()
        plt.plot(x, y1, label=n1)
        plt.plot(x, y2, label=n2)
        plt.xlabel("timestep (index)")
        plt.ylabel(f"obs dim {i}")
        plt.title(f"Obs comparison - dim {i}")
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(dir_path, f"obs_dim_{i}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Plot {i} done.")

# === Main usage ===

dataset_isaac_gym = load_dataset_lazy("expert_dataset.hdf5")
dataset_grand_tour = load_dataset_lazy("offline_dataset_pp.hdf5")

"""
print("Expert Dataset (Isaac Gym)")
for k in list(dataset_isaac_gym.keys()):
    info = f"{k}: {type(dataset_isaac_gym[k])} {dataset_isaac_gym[k].shape}\n"
    print(info.strip())

ep_ret = episode_returns(dataset_isaac_gym["rewards"], dataset_isaac_gym["terminals"])
total_episodes = f"Total episodes: {len(ep_ret)}\n"
median_return = f"Median Episode Return: {np.median(ep_ret)}\n"

print(total_episodes.strip())
print(median_return.strip())
"""

#print_obs_stats(dataset_isaac_gym, dataset_grand_tour)
#print_act_stats(dataset_isaac_gym, dataset_grand_tour)

# create dir and plot actions
acts_plots_dir = "act_plots"
#plot_acts(acts_plots_dir, dataset_isaac_gym, dataset_grand_tour, n1="Isaac Gym", n2="Grand Tour")

obs_plots_dir = "obs_plots"
#plot_obs(obs_plots_dir,dataset_isaac_gym,dataset_grand_tour,n1="Isaac Gym", n2="Grand Tour")


dataset_isaac_gym.close()
dataset_grand_tour.close()
