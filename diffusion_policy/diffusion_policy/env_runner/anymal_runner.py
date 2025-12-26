from typing import Dict, Union
import torch
import torch.nn as nn
from diffusion_policy.env_runner.base_runner import BaseLowdimRunner
from diffusion_policy.policy.base_policy import BaseLowdimPolicy
import sys
import os

# Add parent directory to path to import OnlineEval
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from eval_isaac_v2 import OnlineEval
from isaac_compatibility import make_actions_compatible


class DiffusionPolicyWrapper(nn.Module):
    """Wrapper to make diffusion policy compatible with OnlineEval interface."""
    def __init__(self, policy: BaseLowdimPolicy):
        super().__init__()
        self.policy = policy
    
    @torch.no_grad()
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """For online evaluation - takes single observation and returns action."""
        # observations: [batch, obs_dim] -> need [batch, n_obs_steps, obs_dim]
        if len(observations.shape) == 2:
            # Single timestep, repeat for n_obs_steps
            observations = observations.unsqueeze(1).repeat(
                1, self.policy.n_obs_steps, 1
            )
        
        obs_dict = {'obs': observations}
        result = self.policy.predict_action(obs_dict)
        # Return first action from predicted sequence
        actions = result['action']  # [batch, n_action_steps, action_dim]
        return actions[:, 0]  # Return first action step


class AnymalRunner(BaseLowdimRunner):
    def __init__(self, 
                 output_dir,
                 task_name="anymal_d_flat",
                 seed=27,
                 normalize=False,
                 dataset_path="offline_dataset.hdf5",
                 **kwargs):
        super().__init__(output_dir)
        self.task_name = task_name
        self.seed = seed
        self.normalize = normalize
        self.dataset_path = dataset_path
        self.online_eval = OnlineEval(
            task_name=task_name,
            seed=seed,
            normalize=normalize,
            dataset_path=dataset_path
        )
    
    def run(self, policy: BaseLowdimPolicy) -> Dict:
        """Run online evaluation using OnlineEval.
        
        Returns:
            Dict with evaluation results containing:
            - eval_score: mean episode reward
            - n_episodes: number of episodes evaluated
            - reward_terms: dict of individual reward term averages
            - episode_length: mean episode length
            - obs_stats: dict of observation statistics
        """
        # Wrap policy for OnlineEval interface
        wrapped_policy = DiffusionPolicyWrapper(policy)
        
        # Run evaluation
        result = self.online_eval.eval_actor_isaac(
            actor=wrapped_policy, 
            device=next(policy.parameters()).device
        )
        
        # Return results in format workspace expects
        if len(result) == 5:
            eval_score, n_eps, rew_terms, ep_len, obs_stats = result
        else:
            eval_score, n_eps, rew_terms, ep_len = result
            obs_stats = {}
        
        # Return results dict for workspace to log
        return {
            'eval_score': eval_score,
            'n_episodes': n_eps,
            'reward_terms': rew_terms,
            'episode_length': ep_len,
            'obs_stats': obs_stats
        }

