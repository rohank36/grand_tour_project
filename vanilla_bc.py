# Vanilla Behavioral Cloning (BC) implementation
# Similar structure to cql.py but simplified for pure supervised learning
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pyrallis

import wandb
from tqdm import tqdm
from eval_isaac_v2 import OnlineEval
from utils import compute_mean_std, normalize_states, load_hdf5_dataset
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, TransformedDistribution

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    device: str = "cuda"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 27  # sets seed for online eval
    eval_freq: int = int(1e4)
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(150000)
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    policy_lr: float = 3e-4  # Policy learning rate
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = False  # Normalize states
    policy_log_std_multiplier: float = 1.0  # Stochastic policy std multiplier
    normalize_online_eval_obs: bool = False  # Normalize online eval obs
    dataset_filepath: str = "offline_dataset_pp.hdf5"
    
    project: str = "grand_tour"  # wandb project name
    group: str = "BC"  # wandb group name
    name: str = "BC-vanilla"  # wandb run name

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]


def set_seed(seed, env=None, deterministic_torch=False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthogonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Last layers should be initialized differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
        self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        init_module_weights(self.base_network)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, 
        observations: torch.Tensor, 
        actions: torch.Tensor = None
    ) -> torch.Tensor:
        if actions.ndim == 3 and observations.ndim == 2:
            observations = extend_and_repeat(observations, 1, actions.shape[1])

        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()

        # env-scale -> tanh-space
        if not self.no_tanh:
            actions_tanh = (actions / self.max_action).clamp(-0.999999, 0.999999)
        else:
            actions_tanh = actions

        # This calls ReparameterizedTanhGaussian.log_prob and calculates the log prob of the dataset actions under the current policy distribution
        log_probs = self.tanh_gaussian.log_prob(mean, log_std, actions_tanh)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()
    
    @torch.no_grad()
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            actions, _ = self(observations, deterministic=True)  # deterministic = True for eval
        return actions


class BehavioralCloning:
    def __init__(
        self,
        actor,
        actor_optimizer,
        device: str = "cpu",
    ):
        super().__init__()

        self._device = device
        self.total_it = 0

        self.actor = actor
        self.actor_optimizer = actor_optimizer

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        self.total_it += 1

        # BC loss: negative log-likelihood of actions given observations
        log_probs = self.actor.log_prob(observations, actions)
        policy_loss = -log_probs.mean()

        # Sample actions for logging purposes
        with torch.no_grad():
            new_actions, log_pi = self.actor(observations)
            action_mean = new_actions.mean().item()
            action_std = new_actions.std().item()
            action_min = new_actions.min().item()
            action_max = new_actions.max().item()
            
            # Compute per-dimension action means
            action_mean_per_dim = new_actions.mean(dim=0).cpu().numpy()  # Shape: (action_dim,)
            action_mean_per_dim_dict = {f"action_train/mean_dim_{i}": float(val) for i, val in enumerate(action_mean_per_dim)}
            
            # Log observation stats during training
            obs_mean = observations.mean().item()
            obs_std = observations.std().item()
            obs_min = observations.min().item()
            obs_max = observations.max().item()
            
            # Compute per-dimension observation means
            obs_mean_per_dim = observations.mean(dim=0).cpu().numpy()  # Shape: (obs_dim,)
            obs_mean_per_dim_dict = {f"obs_train/mean_dim_{i}": float(val) for i, val in enumerate(obs_mean_per_dim)}

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            action_mean=action_mean,
            action_std=action_std,
            action_min=action_min,
            action_max=action_max,
            **action_mean_per_dim_dict,  # Add per-dimension action means
            obs_mean=obs_mean,
            obs_std=obs_std,
            obs_min=obs_min,
            obs_max=obs_max,
            **obs_mean_per_dim_dict,  # Add per-dimension observation means
        )

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])
        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    env = None

    # Load dataset
    dataset_path = config.dataset_filepath
    dataset = load_hdf5_dataset(dataset_path)

    state_dim = dataset["observations"].shape[1]
    action_dim = dataset["actions"].shape[1]

    print(f"BC state dim: {state_dim} action dim: {action_dim}")

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(np.max(np.abs(dataset["actions"])))

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed)

    actor = TanhGaussianPolicy(
        state_dim,
        action_dim,
        max_action,
        log_std_multiplier=config.policy_log_std_multiplier,
        orthogonal_init=config.orthogonal_init,
    ).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)

    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "device": config.device,
    }

    print("---------------------------------------")
    print(f"Training BC, Seed: {seed}")
    print("---------------------------------------")

    # Initialize trainer
    trainer = BehavioralCloning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    # -------------------------------------------

    online_eval = OnlineEval(task_name="anymal_d_flat", seed=config.eval_seed, normalize=config.normalize_online_eval_obs)
    start_time = time.time()
    
    for t in tqdm(range(int(config.max_timesteps)), desc="Training BC"):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            result = online_eval.eval_actor_isaac(actor=actor, device=config.device)
            if len(result) == 5:
                eval_score, n_eps_evaluated, scaled_rew_terms_avg, avg_episode_length, obs_stats = result
            else:
                # Backward compatibility
                eval_score, n_eps_evaluated, scaled_rew_terms_avg, avg_episode_length = result
                obs_stats = {}
            
            print("---------------------------------------")
            print(
                f"Evaluation over {n_eps_evaluated} episodes: {eval_score:.3f} "
            )
            print("---------------------------------------")
            
            if config.checkpoints_path:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            
            log_dict_eval = {
                "isaac_reward": eval_score,
                "num_eval_episodes": n_eps_evaluated,
                **{f"reward_terms/{k}": v for k, v in scaled_rew_terms_avg.items()},
                "isaac_avg_episode_length": avg_episode_length,
            }
            # Add observation stats if available - use clear prefix to group Isaac Gym eval metrics
            if obs_stats:
                log_dict_eval.update({f"isaac_eval/{k}": v for k, v in obs_stats.items()})
            
            wandb.log(
                log_dict_eval,
                step=trainer.total_it,
            )

    end_time = time.time()
    training_time_minutes = (end_time - start_time) / 60 
    print(f"\n\n")
    print(f"Training duration: {training_time_minutes:.2f} minutes")

if __name__ == "__main__":
    train()

