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
from grandtour_compatibility import unscale_observations

class OnlineEval:

    def __init__(self, task_name, seed, dataset_path="offline_dataset_pp.hdf5", normalize=False):
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

    
    def calculate_total_reward(self, rewbuffer, ep_infos, lenbuffer):
        """
        Calculate mean reward from episode total rewards buffer and individual reward terms.
        Matches the approach in OnPolicyRunner where we accumulate per-step rewards
        and store episode totals, then compute the mean of episode totals.
        Also computes mean of individual reward terms from episode infos (as normalized values,
        matching OnPolicyRunner which logs them directly without conversion).
        """
        if len(rewbuffer) > 0:
            avg_total_ep_rew = np.mean(rewbuffer)
            avg_episode_length = np.mean(lenbuffer) if len(lenbuffer) > 0 else 0.0
        else:
            avg_total_ep_rew = 0.0
            avg_episode_length = 0.0
        
        # Compute mean of individual reward terms (matching OnPolicyRunner approach exactly, using numpy)
        scaled_rew_terms_avg = {}
        if ep_infos:
            # Process all keys in ep_infos (matching OnPolicyRunner which processes all keys)
            for key in ep_infos[0].keys():
                infotensor = np.array([])
                for ep_info in ep_infos:
                    # Handle scalar and zero dimensional tensor/array infos (exactly like OnPolicyRunner)
                    value = ep_info[key]
                    # Convert torch tensor to numpy if needed
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    # Convert to numpy array if not already
                    if not isinstance(value, np.ndarray):
                        value = np.array([value])
                    # Handle zero dimensional arrays
                    if value.ndim == 0:
                        value = np.expand_dims(value, 0)
                    # Concatenate to infotensor
                    if len(infotensor) == 0:
                        infotensor = value
                    else:
                        infotensor = np.concatenate((infotensor, value))
                # Compute mean (matching OnPolicyRunner's torch.mean)
                value = np.mean(infotensor)
                scaled_rew_terms_avg[key] = float(value)
        
        return avg_total_ep_rew, len(rewbuffer), scaled_rew_terms_avg, avg_episode_length
   
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
        #obs = obs[:, :-12]  # Remove last 12 dimensions (prev_actions)
        
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

        # Track cumulative rewards per environment (matching OnPolicyRunner approach)
        cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
        rewbuffer = []  # Store total episode rewards
        ep_infos = []  # Store episode info dicts for individual reward terms
        lenbuffer = []  # Store episode lengths (matching OnPolicyRunner)

        # Accumulate observation stats for logging
        obs_stats_list = []
        obs_all_list = []  # Store all observations for per-dimension stats
        
        # Accumulate action stats for logging
        actions_all_list = []  # Store all actions for per-dimension stats

        # max_episode_length = 1001
        for i in range(num_repetitions * int(max_episode_length)+2):
        #for i in tqdm(range(num_repetitions * int(max_episode_length)+2),desc="Online IG Eval"):

            obs = unscale_observations(obs, device=device)

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
            #obs = obs[:, :-12]  # Remove last 12 dimensions (prev_actions)

            # Accumulate rewards per environment (matching OnPolicyRunner approach)
            cur_reward_sum += rews.squeeze()

            episode_lengths = torch.clamp(episode_lengths + 1, max=max_episode_length)
            done_indices = torch.where(dones.squeeze() == 1)[0]
            
            # When episodes end, store total episode rewards and reset
            if len(done_indices) > 0:
                rewbuffer.extend(cur_reward_sum[done_indices].cpu().numpy().tolist())
                # Store episode lengths for finished episodes
                lenbuffer.extend(episode_lengths[done_indices].cpu().numpy().tolist())
                cur_reward_sum[done_indices] = 0
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
                # Collect episode info for individual reward terms (matching OnPolicyRunner approach)
                if "episode" in infos and infos["episode"]:
                    ep_infos.append(infos["episode"])
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
            
            eval_score, n_eps_evaluated, scaled_rew_terms_avg, avg_episode_length = self.calculate_total_reward(rewbuffer, ep_infos, lenbuffer)
            
            # Return observation stats along with other eval results
            obs_stats = {
                'isaac_obs_mean': avg_obs_mean,
                'isaac_obs_std': avg_obs_std,
                'isaac_obs_min': avg_obs_min,
                'isaac_obs_max': avg_obs_max,
                **obs_mean_per_dim_dict,  # Add per-dimension observation means
                **actions_mean_per_dim_dict,  # Add per-dimension action means
            }
            
            return eval_score, n_eps_evaluated, scaled_rew_terms_avg, avg_episode_length, obs_stats
        else:
            eval_score, n_eps_evaluated, scaled_rew_terms_avg, avg_episode_length = self.calculate_total_reward(rewbuffer, ep_infos, lenbuffer)
            return eval_score, n_eps_evaluated, scaled_rew_terms_avg, avg_episode_length, {}

        

