""" To make Isaac Gym data compatible with GrandTour during Imitation Learning """

import numpy as np
import torch
from isaac_compatibility import (
    LIN_VEL_SCALE,
    ANG_VEL_SCALE,
    DOF_VEL_SCALE,
    COMMANDS_SCALE,
    obs_scale,
    action_scale,
    build_default_dof_pos,
)

# Isaac Gym observation scaling factors (from legged_robot_config.py)
# These match what Isaac Gym applies in compute_observations()
ISAAC_LIN_VEL_SCALE = 2.0
ISAAC_ANG_VEL_SCALE = 0.25
ISAAC_DOF_POS_SCALE = 1.0
ISAAC_DOF_VEL_SCALE = 0.05
ISAAC_COMMANDS_SCALE = np.array([2.0, 2.0, 0.25])  # [lin_vel_x, lin_vel_y, ang_vel_yaw]


def unscale_lin_vel(lin_vel_scaled):
    """
    Unscale linear velocity from Isaac Gym format to GrandTour format.
    Isaac Gym: lin_vel * 2.0
    GrandTour: lin_vel (unscaled)
    """
    return lin_vel_scaled / ISAAC_LIN_VEL_SCALE


def unscale_ang_vel(ang_vel_scaled):
    """
    Unscale angular velocity from Isaac Gym format to GrandTour format.
    Isaac Gym: ang_vel * 0.25
    GrandTour: ang_vel (unscaled)
    """
    return ang_vel_scaled / ISAAC_ANG_VEL_SCALE


def unscale_commands(commands_scaled):
    """
    Unscale commands from Isaac Gym format to GrandTour format.
    Isaac Gym: commands * [2.0, 2.0, 0.25]
    GrandTour: commands (unscaled)
    """
    if isinstance(commands_scaled, torch.Tensor):
        return commands_scaled / torch.tensor(ISAAC_COMMANDS_SCALE, device=commands_scaled.device, dtype=commands_scaled.dtype)
    return commands_scaled / ISAAC_COMMANDS_SCALE


def unscale_joint_pos(joint_pos_scaled):
    """
    Unscale joint positions from Isaac Gym format to GrandTour format.
    Isaac Gym: (dof_pos - default_dof_pos) * 1.0
    GrandTour: joint_pos (absolute positions, unscaled)
    """
    default_dof_pos = build_default_dof_pos()
    
    if isinstance(joint_pos_scaled, torch.Tensor):
        default_dof_pos_torch = torch.tensor(default_dof_pos, device=joint_pos_scaled.device, dtype=joint_pos_scaled.dtype)
        # Isaac Gym stores: (dof_pos - default_dof_pos) * scale
        # To get absolute: (scaled_offset / scale) + default_dof_pos
        return (joint_pos_scaled / ISAAC_DOF_POS_SCALE) + default_dof_pos_torch
    
    return (joint_pos_scaled / ISAAC_DOF_POS_SCALE) + default_dof_pos


def unscale_joint_vel(joint_vel_scaled):
    """
    Unscale joint velocities from Isaac Gym format to GrandTour format.
    Isaac Gym: dof_vel * 0.05
    GrandTour: joint_vel (unscaled)
    """
    return joint_vel_scaled / ISAAC_DOF_VEL_SCALE


def unscale_previous_actions(actions_scaled):
    """
    Convert previous actions from Isaac Gym format (offsets) to GrandTour format (absolute positions).
    Isaac Gym stores actions as normalized offsets: action (offset from default_dof_pos)
    GrandTour stores actions as absolute positions
    
    This is the inverse of make_actions_compatible():
    - make_actions_compatible: absolute -> (absolute - default) / action_scale
    - unscale_previous_actions: offset -> offset * action_scale + default
    
    """
    default_dof_pos = build_default_dof_pos()
    
    if isinstance(actions_scaled, torch.Tensor):
        default_dof_pos_torch = torch.tensor(default_dof_pos, device=actions_scaled.device, dtype=actions_scaled.dtype)
        # Convert offset back to absolute: offset * action_scale + default_dof_pos
        return actions_scaled * action_scale + default_dof_pos_torch
    
    # Convert offset back to absolute: offset * action_scale + default_dof_pos
    return actions_scaled * action_scale + default_dof_pos


def unscale_observations(obs_isaac, device="cpu"):
    """
    Unscale full observation vector from Isaac Gym format to GrandTour format.
    
    Observation structure (48 dims, no height measurements):
    - [0:3]: base_lin_vel (scaled by 2.0)
    - [3:6]: base_ang_vel (scaled by 0.25)
    - [6:9]: projected_gravity (no scaling)
    - [9:12]: commands (scaled by [2.0, 2.0, 0.25])
    - [12:24]: dof_pos (scaled as (dof_pos - default_dof_pos) * 1.0)
    - [24:36]: dof_vel (scaled by 0.05)
    - [36:48]: previous actions (stored as offsets, need to convert to absolute positions)
    
    Args:
        obs_isaac: Observations from Isaac Gym (torch.Tensor or np.ndarray)
        device: Device for torch tensors (if input is torch.Tensor)
    
    Returns:
        Observations in GrandTour format (same type as input)
    """
    is_torch = isinstance(obs_isaac, torch.Tensor)
    
    if is_torch:
        obs_gt = obs_isaac.clone()
    else:
        obs_gt = obs_isaac.copy()
    
    # Unscale base_lin_vel [0:3]
    obs_gt[..., 0:3] = unscale_lin_vel(obs_gt[..., 0:3])
    
    # Unscale base_ang_vel [3:6]
    obs_gt[..., 3:6] = unscale_ang_vel(obs_gt[..., 3:6])
    
    # projected_gravity [6:9] - no scaling needed
    
    # Unscale commands [9:12]
    obs_gt[..., 9:12] = unscale_commands(obs_gt[..., 9:12])
    
    # Unscale dof_pos [12:24]
    obs_gt[..., 12:24] = unscale_joint_pos(obs_gt[..., 12:24])
    
    # Unscale dof_vel [24:36]
    obs_gt[..., 24:36] = unscale_joint_vel(obs_gt[..., 24:36])
    
    # Unscale previous actions [36:48] - convert from offsets to absolute positions
    obs_gt[..., 36:48] = unscale_previous_actions(obs_gt[..., 36:48])
    
    return obs_gt
