import numpy as np
from typing import Tuple, Dict
import h5py 

# wrote in a independent file to use the same code in both online eval and cql

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

def load_hdf5_dataset(path: str) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        dataset = {
            "observations":      f["observations"][()],
            "next_observations": f["next_observations"][()],
            "actions":           f["actions"][()],
            "rewards":           f["rewards"][()],
            "terminals":         f["terminals"][()],
        }
    return dataset