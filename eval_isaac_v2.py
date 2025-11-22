import argparse
import sys

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym

from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, export_policy_as_onnx
import numpy as np
import torch


@torch.no_grad()
def eval_actor_isaac(
    actor: torch.nn.Module,
    task_name: str = "anymal_c_flat",
    num_envs: int = 1,
    n_episodes: int = 10,
    device: str = "cuda",
) -> np.ndarray:
    
    num_repetitions = 3
    args = get_args()
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
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    num_envs = env_cfg.env.num_envs
    max_episode_length = int(env.max_episode_length)

    episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=env.device)

    collected_episode_rewards = []

    for i in range(num_repetitions * int(max_episode_length)):
        actions = actor.act_inference(obs.detach())
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

        if "episode" in infos:
            # Check if any episodes finished in this step
            if "r" in infos["episode"]:
                rewards = infos["episode"]["r"]
                # Convert tensor to numpy and extend list
                if isinstance(rewards, torch.Tensor):
                    rewards = rewards.cpu().numpy()
                collected_episode_rewards.extend(rewards)

        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
                    
        elif i == stop_rew_log:
            logger.print_rewards()

    torch.cuda.synchronize()  # Ensure all CUDA operations are completed

    return np.array(collected_episode_rewards)

        

