import argparse
import sys

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym

from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, export_policy_as_onnx
import numpy as np
import torch

from tqdm import tqdm

from reward import rewards
from utils import compute_mean_std, load_hdf5_dataset

class OnlineEval:

    def __init__(self, task_name, seed, dataset_path="offline_dataset_pp.hdf5", normalize=True):
        args = get_args()
        args.seed = seed
        args.task = task_name
        args.headless = True # Set headless for faster eval
        env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)

        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False
        env_cfg.noise.add_noise = False

        if args.domain_rand_friction_range is not None:
            env_cfg.domain_rand.randomize_friction = True
            env_cfg.domain_rand.randomize_added_mass = False
            env_cfg.domain_rand.push_robots = False
            env_cfg.domain_rand.randomize_ground_friction = False
            env_cfg.domain_rand.friction_range = args.domain_rand_friction_range
        elif args.domain_rand_added_mass_range is not None:
            env_cfg.domain_rand.randomize_added_mass = True
            env_cfg.domain_rand.randomize_friction = False
            env_cfg.domain_rand.push_robots = False
            env_cfg.domain_rand.randomize_ground_friction = False
            env_cfg.domain_rand.added_mass_range = args.domain_rand_added_mass_range
        elif args.domain_rand_push_range is not None:
            env_cfg.domain_rand.push_robots = True
            env_cfg.domain_rand.randomize_friction = False
            env_cfg.domain_rand.randomize_added_mass = False
            env_cfg.domain_rand.randomize_ground_friction = False
            # env_cfg.domain_rand.push_interval_s = 10
            env_cfg.domain_rand.max_push_vel_xy = args.domain_rand_push_range
        elif args.domain_rand_ground_friction_range is not None:
            env_cfg.domain_rand.randomize_ground_friction = True
            env_cfg.domain_rand.randomize_friction = False
            env_cfg.domain_rand.randomize_added_mass = False
            env_cfg.domain_rand.push_robots = False
            env_cfg.domain_rand.ground_friction_range = args.domain_rand_ground_friction_range


        # prepare environment
        #env_cfg.env.num_envs = 1# FOR TESTING
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

        self.env_cfg = env_cfg
        self.env = env
        self.args = args

        self.scales_dict = {
            name: value
            for name, value in vars(rewards.scales).items()
        }

        decimation = 4
        physics_base = 0.005
        self.dt = decimation * physics_base

        # Load normalization stats from the same dataset used for training
        self.normalize = normalize
        if self.normalize:
            dataset = load_hdf5_dataset(dataset_path)
            self.state_mean, self.state_std = compute_mean_std(dataset["observations"], eps=1e-3)

    """
    def calculate_total_reward(self,logger: Logger):
        avg_total_ep_rew = 0
        scaled_rew_terms_avg = {}
        for key,values in logger.rew_log.items():
            values_scaled = np.array(values) * self.scales_dict[key.replace("rew_", "")] * self.dt
            mean = np.sum(values_scaled) / logger.num_episodes
            scaled_rew_terms_avg[key] = mean
            avg_total_ep_rew += mean
        
        return avg_total_ep_rew, logger.num_episodes, scaled_rew_terms_avg
    """
    def calculate_total_reward(self, logger: Logger):
        avg_total_ep_rew = 0
        scaled_rew_terms_avg = {}
        for key, values in logger.rew_log.items():
            mean_per_sec = np.sum(np.array(values)) / logger.num_episodes
            # Convert to total episode reward
            mean_total = mean_per_sec * self.env.max_episode_length_s
            scaled_rew_terms_avg[key] = mean_total
            avg_total_ep_rew += mean_total
        
        return avg_total_ep_rew, logger.num_episodes, scaled_rew_terms_avg

    @torch.no_grad()
    def eval_actor_isaac(
        self,
        actor: torch.nn.Module,
        #task_name: str = "anymal_c_flat",
        device: str = "cuda",
    ) -> np.ndarray:
        
        actor.eval()

        num_repetitions = 1
        

        env = self.env
        env_cfg = self.env_cfg

        obs = env.get_observations()
        
        logger = Logger(env.dt) # note env.dt = 0.0199999 
        robot_index = 0  # which robot is used for logging
        joint_index = 1  # which joint is used for logging
        stop_state_log = 1000  # number of steps before plotting states
        stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
        camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
        camera_vel = np.array([1., 1., 0.])
        camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
        img_idx = 0

        num_envs = env_cfg.env.num_envs # normally 4096
        max_episode_length = int(env.max_episode_length)

        episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=env.device)

        # Convert normalization stats to torch tensors on the correct device
        if self.normalize:
            state_mean_torch = torch.tensor(self.state_mean, dtype=torch.float32, device=device)
            state_std_torch = torch.tensor(self.state_std, dtype=torch.float32, device=device)

        # Accumulate observation stats for logging
        obs_stats_list = []
        obs_all_list = []  # Store all observations for per-dimension stats
        
        # Accumulate action stats for logging
        actions_all_list = []  # Store all actions for per-dimension stats

        # max_episode_length = 1001
        for i in range(num_repetitions * int(max_episode_length)+2):
        #for i in tqdm(range(num_repetitions * int(max_episode_length)+2),desc="Online IG Eval"):
            # Normalize observations before feeding to actor
            if self.normalize:
                obs_normalized = (obs - state_mean_torch) / state_std_torch
            else:
                obs_normalized = obs
            
            # Collect observation stats (from raw Isaac Gym observations)
            with torch.no_grad():
                obs_stats_list.append({
                    'mean': obs.mean().item(),
                    'std': obs.std().item(),
                    'min': obs.min().item(),
                    'max': obs.max().item(),
                })
                # Store observations for per-dimension analysis
                obs_all_list.append(obs.cpu().numpy())
            
            actions = actor.act_inference(obs_normalized.detach())
            # Store actions for per-dimension analysis
            with torch.no_grad():
                actions_all_list.append(actions.cpu().numpy())
            obs, _, rews, dones, infos = env.step(actions.detach())

            episode_lengths = torch.clamp(episode_lengths + 1, max=max_episode_length)
            done_indices = torch.where(dones.squeeze() == 1)[0]
            for env_idx in done_indices:
                env_idx = env_idx.item()
                episode_lengths[env_idx] = 0

            
            if i < stop_state_log:
                logger.log_states(
                    {
                        'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                        'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                        'dof_torque': env.torques[robot_index, joint_index].item(),
                        'command_x': env.commands[robot_index, 0].item(),
                        'command_y': env.commands[robot_index, 1].item(),
                        'command_yaw': env.commands[robot_index, 2].item(),
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                        'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                    }
                )
            #elif i == stop_state_log:
                #logger.plot_states(model_dir)

            if 0 < i < stop_rew_log:
                #if infos["episode"]:
                if "episode" in infos and infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes > 0:
                        logger.log_rewards(infos["episode"], num_episodes)
                        
            #elif i == stop_rew_log:
                #logger.print_rewards()

        torch.cuda.synchronize()  # Ensure all CUDA operations are completed
        actor.train() 

        # Compute average observation stats across all steps
        if obs_stats_list:
            avg_obs_mean = np.mean([s['mean'] for s in obs_stats_list])
            avg_obs_std = np.mean([s['std'] for s in obs_stats_list])
            avg_obs_min = np.mean([s['min'] for s in obs_stats_list])
            avg_obs_max = np.mean([s['max'] for s in obs_stats_list])
            
            # Compute per-dimension mean from all collected observations
            if obs_all_list:
                obs_all_array = np.concatenate(obs_all_list, axis=0)  # Shape: (total_steps * num_envs, obs_dim)
                obs_mean_per_dim = obs_all_array.mean(axis=0)  # Shape: (obs_dim,)
                # Create dict with per-dimension means
                obs_mean_per_dim_dict = {f'isaac_obs_mean_dim_{i}': float(val) for i, val in enumerate(obs_mean_per_dim)}
            else:
                obs_mean_per_dim_dict = {}
            
            # Compute per-dimension mean from all collected actions
            if actions_all_list:
                actions_all_array = np.concatenate(actions_all_list, axis=0)  # Shape: (total_steps * num_envs, action_dim)
                actions_mean_per_dim = actions_all_array.mean(axis=0)  # Shape: (action_dim,)
                # Create dict with per-dimension means
                actions_mean_per_dim_dict = {f'isaac_action_mean_dim_{i}': float(val) for i, val in enumerate(actions_mean_per_dim)}
            else:
                actions_mean_per_dim_dict = {}
            
            eval_score, n_eps_evaluated, scaled_rew_terms_avg = self.calculate_total_reward(logger)
            
            # Return observation stats along with other eval results
            obs_stats = {
                'isaac_obs_mean': avg_obs_mean,
                'isaac_obs_std': avg_obs_std,
                'isaac_obs_min': avg_obs_min,
                'isaac_obs_max': avg_obs_max,
                **obs_mean_per_dim_dict,  # Add per-dimension observation means
                **actions_mean_per_dim_dict,  # Add per-dimension action means
            }
            
            return eval_score, n_eps_evaluated, scaled_rew_terms_avg, obs_stats
        else:
            eval_score, n_eps_evaluated, scaled_rew_terms_avg = self.calculate_total_reward(logger)
            return eval_score, n_eps_evaluated, scaled_rew_terms_avg, {}

        

