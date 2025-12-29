# Grand Tour Offline RL

<div align="center">

**Offline Reinforcement Learning for Quadruped Locomotion using the Grand Tour Dataset**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)

</div>

---

## Overview

This project implements **offline reinforcement learning algorithms** for training quadruped locomotion policies on the [Grand Tour Dataset](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset) — a large-scale real-world dataset of ANYmal D robot trajectories. The trained policies are evaluated in the **Isaac Gym** physics simulator.

### Key Features

- 🤖 **Multiple Offline RL Algorithms**: Vanilla BC, IQL, CQL, EDAC
- 🎯 **Diffusion Policy Support**: Transformer-based diffusion policies for locomotion
- 🔄 **End-to-end Pipeline**: From raw sensor data to trained policies
- 📊 **Isaac Gym Integration**: Online evaluation in high-fidelity simulation
- 📈 **Weights & Biases Logging**: Comprehensive experiment tracking

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Grand Tour Dataset                               │
│                    (Real-world ANYmal D trajectories)                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │   download   │───▶│  align_data  │───▶│    build_dataset         │   │
│  │    .py       │    │     .py      │    │        .py               │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Offline Dataset (HDF5)                              │
│         observations, actions, rewards, terminals, next_obs              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │  Vanilla BC  │ │     IQL      │ │    CQL       │
           │              │ │              │ │              │
           └──────────────┘ └──────────────┘ └──────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Online Evaluation (Isaac Gym)                         │
│              eval_isaac_v2.py → ANYmal D Flat Terrain                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.x or 12.x
- Isaac Gym Preview 4
- Hugging Face account (for dataset access)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/grand_tour_project.git
   cd grand_tour_project/grand_tour_project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install Isaac Gym**
   ```bash
   # Download Isaac Gym Preview 4 from NVIDIA
   cd /path/to/isaacgym/python
   pip install -e .
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install legged_gym (ANYmal D support)**
   ```bash
   pip install -e git+https://github.com/modanesh/legged_gym_anymal_d.git#egg=legged_gym
   ```

6. **Configure Hugging Face token**
   ```bash
   # Create .env file
   echo "HF_TOKEN=your_huggingface_token" > .env
   ```

---

## Data Pipeline

### 1. Download Grand Tour Dataset

Download real-world ANYmal D trajectories from Hugging Face:

```bash
python download.py
```

This downloads the following sensor topics:
- `anymal_state_odometry` — Robot pose and velocity
- `anymal_state_state_estimator` — Joint positions, velocities, contact states
- `anymal_imu` — IMU measurements
- `anymal_state_actuator` — Actuator commands
- `anymal_command_twist` — Velocity commands

### 2. Align Sensor Data

Align multi-sensor data to a common 50Hz timeline:

```bash
python align_data.py
```

Output: `aligned_data.zarr` containing synchronized sensor readings.

### 3. Build Offline RL Dataset

Convert aligned data to offline RL format compatible with Isaac Gym:

```bash
python build_dataset.py
```

**Output**: `offline_dataset_pp.hdf5` with:
- `observations` — 36D state vectors (or 48D with previous actions)
- `actions` — 12D joint position targets
- `rewards` — Computed using Isaac Gym reward functions
- `terminals` — Episode boundaries
- `next_observations` — Next state

### Observation Space (36 dimensions)

| Index | Description | Scale |
|-------|-------------|-------|
| 0-2 | Base linear velocity (body frame) | × 2.0 |
| 3-5 | Base angular velocity (body frame) | × 0.25 |
| 6-8 | Projected gravity | — |
| 9-11 | Commands [vx, vy, ω_yaw] | × [2.0, 2.0, 0.25] |
| 12-23 | Joint positions (offset from default) | × 1.0 |
| 24-35 | Joint velocities | × 0.05 |

### Action Space (12 dimensions)

Joint position targets for the 12 DOF:
```
LF_HAA, LF_HFE, LF_KFE, RF_HAA, RF_HFE, RF_KFE,
LH_HAA, LH_HFE, LH_KFE, RH_HAA, RH_HFE, RH_KFE
```

---

## Training

### Vanilla Behavioral Cloning (BC)

Simple supervised learning to imitate expert demonstrations:

```bash
python vanilla_bc.py --dataset_filepath=offline_dataset_pp.hdf5 \
                     --max_timesteps=200000 \
                     --hidden_dim=512 \
                     --n_hidden_layers=3
```

**Key hyperparameters:**
- `hidden_dim`: Network hidden layer size (default: 512)
- `n_hidden_layers`: Number of hidden layers (default: 3)
- `learning_rate`: Learning rate (default: 1e-3)
- `normalize`: Normalize observations (default: False)

### Implicit Q-Learning (IQL)

Offline RL with implicit Q-function learning:

```bash
python iql.py --dataset_filepath=offline_dataset_pp.hdf5 \
              --max_timesteps=500000 \
              --beta=3.0 \
              --iql_tau=0.7
```

**Key hyperparameters:**
- `beta`: Inverse temperature for advantage weighting (default: 3.0)
- `iql_tau`: Asymmetric loss coefficient (default: 0.7)
- `discount`: Discount factor (default: 0.99)

### Conservative Q-Learning (CQL)

Offline RL with conservative value estimation:

```bash
python cql.py --dataset_filepath=offline_dataset_pp.hdf5 \
              --max_timesteps=150000 \
              --cql_alpha=5.0
```

**Key hyperparameters:**
- `cql_alpha`: CQL regularization weight (default: 5.0)
- `bc_steps`: Initial BC pretraining steps (default: 150000)
- `cql_n_actions`: Number of sampled actions (default: 10)

### Ensemble Diversified Actor Critic (EDAC)

SAC-based offline RL with ensemble diversification:

```bash
python edac.py --dataset_filepath=offline_dataset_pp.hdf5 \
               --num_epochs=1500 \
               --num_critics=10 \
               --eta=1.0
```

**Key hyperparameters:**
- `num_critics`: Size of critic ensemble (default: 10)
- `eta`: Ensemble diversification coefficient (default: 1.0)
- `tau`: Target network update rate (default: 5e-3)

### Diffusion Policy

Train transformer-based diffusion policy:

```bash
python diffuseloco.py --config-name=anymal_diffusion_policy
```

Configuration files are in `diffusion_policy/config_files/`.

---

## Evaluation

All training scripts automatically evaluate in Isaac Gym every `eval_freq` steps.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `isaac_reward` | Total episode reward |
| `isaac_avg_episode_length` | Average episode length |
| `reward_terms/tracking_lin_vel` | Linear velocity tracking reward |
| `reward_terms/tracking_ang_vel` | Angular velocity tracking reward |
| `reward_terms/feet_air_time` | Feet air time reward |
| `reward_terms/action_rate` | Action smoothness penalty |

### Reward Functions

The reward computation follows the Isaac Gym ANYmal configuration:

```python
reward_scales = {
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.5,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "torques": -0.00001,
    "dof_acc": -2.5e-7,
    "feet_air_time": 1.0,
    "action_rate": -0.01,
}
```

---

## Project Structure

```
grand_tour_project/
├── download.py              # Download Grand Tour dataset from HuggingFace
├── align_data.py            # Align multi-sensor data to 50Hz
├── build_dataset.py         # Build offline RL dataset (HDF5)
├── build_dataset_full_pipeline.py  # End-to-end dataset building
│
├── vanilla_bc.py            # Vanilla Behavioral Cloning
├── iql.py                   # Implicit Q-Learning
├── cql.py                   # Conservative Q-Learning
├── edac.py                  # Ensemble Diversified Actor Critic
│
├── eval_isaac_v2.py         # Isaac Gym online evaluation
├── reward.py                # Reward function implementations
├── utils.py                 # Utility functions
│
├── isaac_compatibility.py   # Grand Tour → Isaac Gym data transforms
├── grandtour_compatibility.py # Isaac Gym → Grand Tour data transforms
│
├── diffusion_policy/        # Diffusion policy implementation
│   ├── config/              # Hydra configuration files
│   ├── dataset/             # Dataset loaders
│   ├── model/               # Network architectures
│   ├── policy/              # Policy implementations
│   └── workspace/           # Training workspaces
│
├── diffuseloco.py           # Diffusion policy training entry point
├── compare_dataset_*.py     # Dataset analysis scripts
│
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## Configuration

### Common Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--device` | Training device | `cuda` |
| `--seed` | Random seed | `0` |
| `--eval_seed` | Evaluation seed | `27` |
| `--eval_freq` | Evaluation frequency | `10000` |
| `--batch_size` | Training batch size | `256` |
| `--checkpoints_path` | Model save directory | `None` |
| `--dataset_filepath` | Path to HDF5 dataset | `offline_dataset_pp.hdf5` |
| `--include_prev_actions` | Include previous actions in obs (48D) | `False` |
| `--normalize` | Normalize observations | `False` |

### Weights & Biases Integration

All training scripts log to W&B:

```python
project: "grand_tour"
group: "IQL" / "CQL" / "VanillaBC" / "EDAC"
```

View experiments at [wandb.ai](https://wandb.ai).

---

## Data Format Compatibility

### Observation Scaling

The project handles observation scaling between Grand Tour (raw) and Isaac Gym (scaled) formats:

| Component | Grand Tour | Isaac Gym Scale |
|-----------|------------|-----------------|
| Linear velocity | Raw | × 2.0 |
| Angular velocity | Raw | × 0.25 |
| Commands | Raw | × [2.0, 2.0, 0.25] |
| Joint positions | Absolute | Offset from default |
| Joint velocities | Raw | × 0.05 |

### Action Conversion

Actions are converted between formats using:
- **Grand Tour**: Absolute joint positions
- **Isaac Gym**: Normalized offsets from default positions

```python
# Grand Tour → Isaac Gym
action_isaac = (action_gt - default_dof_pos) / action_scale

# Isaac Gym → Grand Tour  
action_gt = action_isaac * action_scale + default_dof_pos
```

---

## Citation

If you use this code, please cite the Grand Tour dataset:

```bibtex
@article{grand_tour_2024,
  title={Grand Tour: A Large-Scale Dataset of Quadruped Robot Trajectories},
  author={Leggedrobotics},
  year={2024},
  publisher={Hugging Face}
}
```

And the relevant algorithm papers:

<details>
<summary>IQL</summary>

```bibtex
@article{kostrikov2021iql,
  title={Offline Reinforcement Learning with Implicit Q-Learning},
  author={Kostrikov, Ilya and Nair, Ashvin and Levine, Sergey},
  journal={arXiv preprint arXiv:2110.06169},
  year={2021}
}
```
</details>

<details>
<summary>CQL</summary>

```bibtex
@article{kumar2020cql,
  title={Conservative Q-Learning for Offline Reinforcement Learning},
  author={Kumar, Aviral and Zhou, Aurick and Tucker, George and Levine, Sergey},
  journal={NeurIPS},
  year={2020}
}
```
</details>

<details>
<summary>EDAC</summary>

```bibtex
@article{an2021edac,
  title={Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble},
  author={An, Gaon and Moon, Seungyong and Kim, Jang-Hyun and Song, Hyun Oh},
  journal={NeurIPS},
  year={2021}
}
```
</details>

---

## Acknowledgments

- [Grand Tour Dataset](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset) by Leggedrobotics
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) by NVIDIA
- [legged_gym_anymal_d](https://github.com/modanesh/legged_gym_anymal_d) for ANYmal D support
- [CORL](https://github.com/tinkoff-ai/CORL) for offline RL algorithm implementations

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

