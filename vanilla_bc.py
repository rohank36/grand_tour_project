import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from eval_isaac_v2 import OnlineEval
from utils import compute_mean_std, normalize_states, load_hdf5_dataset
import time

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    device: str = "cuda"
    seed: int = 0  # Sets PyTorch and Numpy seeds
    eval_seed: int = 27  # sets seed for online eval
    eval_freq: int = int(1e4)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(150000)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    learning_rate: float = 1e-3  # Learning rate
    weight_decay: float = 1e-4  # L2 regularization
    normalize: bool = False  # Normalize states
    normalize_online_eval_obs: bool = False  # Normalize online eval obs
    dataset_filepath: str = "offline_dataset_pp.hdf5"
    
    # Model architecture
    hidden_dim: int = 512  # Hidden layer dimension
    n_hidden_layers: int = 3  # Number of hidden layers
    dropout: float = 0.0  # Dropout rate (0 = disabled)
    
    # Learning rate scheduler
    lr_scheduler: str = "none"  # Options: "none", "cosine", "step", "exponential"
    lr_scheduler_gamma: float = 0.9  # For step/exponential schedulers
    lr_scheduler_step_size: int = 50000  # For step scheduler
    lr_scheduler_T_max: int = 150000  # For cosine scheduler
    
    project: str = "grand_tour"  # wandb project name
    group: str = "VanillaBC"  # wandb group name
    name: str = "VanillaBC"  # wandb run name

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


class VanillaBC(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=512, n_hidden_layers=3, dropout=0.0):
        super(VanillaBC, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, action_dim))
        # Add nn.Tanh() here if actions are normalized to [-1, 1]
        
        self.net = nn.Sequential(*layers)

    def forward(self, obs_history):
        # obs_history: [Batch, State_Dim]
        return self.net(obs_history)
    
    @torch.no_grad()
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """For online evaluation - takes observations and returns actions"""
        with torch.no_grad():
            actions = self(observations)
        return actions


@pyrallis.wrap()
def train(config: TrainConfig):
    # Load dataset
    dataset_path = config.dataset_filepath
    dataset = load_hdf5_dataset(dataset_path)

    state_dim = dataset["observations"].shape[1]
    action_dim = dataset["actions"].shape[1]

    print(f"VanillaBC state dim: {state_dim} action dim: {action_dim}")

    # Normalize states if configured
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
    
    # Create replay buffer and load dataset
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

    # Initialize model, loss, and optimizer
    model = VanillaBC(
        input_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        n_hidden_layers=config.n_hidden_layers,
        dropout=config.dropout
    ).to(config.device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Initialize learning rate scheduler
    if config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.lr_scheduler_T_max
        )
    elif config.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.lr_scheduler_step_size, gamma=config.lr_scheduler_gamma
        )
    elif config.lr_scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.lr_scheduler_gamma
        )
    else:
        scheduler = None

    # Load model if specified
    if config.load_model != "":
        checkpoint = torch.load(config.load_model)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded model from {config.load_model}")

    # Initialize wandb
    wandb_init(asdict(config))

    # Initialize online evaluation
    online_eval = OnlineEval(
        task_name="anymal_d_flat",
        seed=config.eval_seed,
        normalize=config.normalize_online_eval_obs,
    )
    
    start_time = time.time()
    total_it = 0
    
    # Training loop
    for t in tqdm(range(int(config.max_timesteps)), desc="Training VanillaBC"):
        # Sample batch
        batch = replay_buffer.sample(config.batch_size)
        states, actions, rewards, next_states, dones = batch
        states = states.to(config.device)
        actions = actions.to(config.device)

        # Forward pass
        predicted_actions = model(states)
        loss = criterion(predicted_actions, actions)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Step learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        total_it += 1

        # Log training metrics
        with torch.no_grad():
            action_mean = predicted_actions.mean().item()
            action_std = predicted_actions.std().item()
            action_min = predicted_actions.min().item()
            action_max = predicted_actions.max().item()

            # Compute per-dimension action means
            action_mean_per_dim = predicted_actions.mean(dim=0).cpu().numpy()
            action_mean_per_dim_dict = {
                f"action_train/mean_dim_{i}": float(val)
                for i, val in enumerate(action_mean_per_dim)
            }

            # Log observation stats during training
            obs_mean = states.mean().item()
            obs_std = states.std().item()
            obs_min = states.min().item()
            obs_max = states.max().item()

            # Compute per-dimension observation means
            obs_mean_per_dim = states.mean(dim=0).cpu().numpy()
            obs_mean_per_dim_dict = {
                f"obs_train/mean_dim_{i}": float(val)
                for i, val in enumerate(obs_mean_per_dim)
            }

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        
        log_dict = {
            "loss": loss.item(),
            "learning_rate": current_lr,
            "action_mean": action_mean,
            "action_std": action_std,
            "action_min": action_min,
            "action_max": action_max,
            **action_mean_per_dim_dict,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "obs_min": obs_min,
            "obs_max": obs_max,
            **obs_mean_per_dim_dict,
        }

        wandb.log(log_dict, step=total_it)

        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            result = online_eval.eval_actor_isaac(actor=model, device=config.device)
            if len(result) == 5:
                (
                    eval_score,
                    n_eps_evaluated,
                    scaled_rew_terms_avg,
                    avg_episode_length,
                    obs_stats,
                ) = result
            else:
                # Backward compatibility
                (
                    eval_score,
                    n_eps_evaluated,
                    scaled_rew_terms_avg,
                    avg_episode_length,
                ) = result
                obs_stats = {}

            print("---------------------------------------")
            print(
                f"Evaluation over {n_eps_evaluated} episodes: {eval_score:.3f} "
            )
            print("---------------------------------------")

            if config.checkpoints_path:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "total_it": total_it,
                }
                if scheduler is not None:
                    checkpoint["scheduler_state_dict"] = scheduler.state_dict()
                torch.save(
                    checkpoint,
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

            log_dict_eval = {
                "isaac_reward": eval_score,
                "num_eval_episodes": n_eps_evaluated,
                **{f"reward_terms/{k}": v for k, v in scaled_rew_terms_avg.items()},
                "isaac_avg_episode_length": avg_episode_length,
            }
            # Add observation stats if available
            if obs_stats:
                log_dict_eval.update(
                    {f"isaac_eval/{k}": v for k, v in obs_stats.items()}
                )

            wandb.log(log_dict_eval, step=total_it)

    end_time = time.time()
    training_time_minutes = (end_time - start_time) / 60
    print(f"\n\n")
    print(f"Training duration: {training_time_minutes:.2f} minutes")


if __name__ == "__main__":
    train()
