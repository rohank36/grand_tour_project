from typing import Dict
import torch
import numpy as np
import h5py
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class AnymalDataset(BaseLowdimDataset):
    def __init__(self, 
            hdf5_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            state_key='state',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        
        # Load HDF5 data
        with h5py.File(hdf5_path, 'r') as f:
            observations = f['observations'][:]
            actions = f['actions'][:]
            terminals = f['terminals'][:]
        
        # Convert to ReplayBuffer format
        # Compute episode_ends from terminals
        terminal_indices = np.where(terminals)[0]
        episode_ends = []
        cumulative = 0
        for term_idx in terminal_indices:
            cumulative = term_idx + 1
            episode_ends.append(cumulative)
        if len(episode_ends) == 0 or episode_ends[-1] < len(observations):
            episode_ends.append(len(observations))
        episode_ends = np.array(episode_ends, dtype=np.int64)
        
        # Create ReplayBuffer from numpy dict
        root = {
            'data': {
                state_key: observations,
                action_key: actions
            },
            'meta': {
                'episode_ends': episode_ends
            }
        }
        self.replay_buffer = ReplayBuffer(root=root)
        
        # Setup train/val split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    
    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def _sample_to_data(self, sample):
        obs = sample[self.state_key]
        data = {
            'obs': obs,  # T, D_o
            'action': sample[self.action_key][:],  # T, D_a
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

