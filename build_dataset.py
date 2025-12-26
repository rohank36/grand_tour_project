from pathlib import Path
import warnings
import zarr
warnings.filterwarnings('ignore')
import numpy as np
from reward import compute_rewards_offline
import h5py
from isaac_compatibility import *

# Dataset configuration class
class DatasetConfig:
    scale_lin_vel = True
    scale_ang_vel = True
    scale_commands = True
    scale_joint_pos = True
    scale_joint_vel = True
    scale_actions = True


# build offline rl dataset 

def get_axis_params(value, axis_idx):
    axis = np.zeros(3)
    axis[axis_idx] = value
    return axis

def quat_rotate_inverse(q, v):
    """
    Rotate vector(s) v by the inverse of quaternion(s) q.
    Matches Isaac Gym's PyTorch implementation.
    q: (..., 4) array [x, y, z, w]
    v: (..., 3) array
    returns: rotated v in same shape
    """
    q = np.asarray(q)
    v = np.asarray(v)

    q_vec = q[..., :3]         # (x, y, z)
    q_w = q[..., 3]            # w
    
    # Match PyTorch implementation:
    # a = v * (2.0 * q_w^2 - 1.0)
    # b = 2.0 * q_w * cross(q_vec, v)
    # c = 2.0 * q_vec * (q_vec · v)
    # return a - b + c
    
    a = v * (2.0 * q_w[..., None] ** 2 - 1.0)
    b = 2.0 * q_w[..., None] * np.cross(q_vec, v)
    # Dot product: q_vec · v, then broadcast to multiply with q_vec
    dot_product = np.sum(q_vec * v, axis=-1, keepdims=True)  # (..., 1)
    c = 2.0 * q_vec * dot_product
    return a - b + c

def build_offline_dataset(data, cfg=DatasetConfig(), episode_len_s=20, hz=50):
    """Convert aligned ANYmal sensor data into offline RL dataset."""

    est = data["sensors"]["anymal_state_state_estimator"]
    act = data["sensors"]["anymal_state_actuator"]
    cmd = data["sensors"]["anymal_command_twist"]
    imu = data["sensors"]["anymal_imu"]
    odom = data["sensors"]["anymal_state_odometry"]

    # Construct root_states from odometry data (all in world frame)
    # root_states structure: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13 dims
    root_states = np.concatenate([
        odom["pose_pos"],        # [0:3] Position [x, y, z] in world frame
        odom["pose_orien"],      # [3:7] Quaternion [x, y, z, w] in world frame
        odom["twist_lin"],       # [7:10] Linear Velocity [vx, vy, vz] in world frame
        odom["twist_ang"],       # [10:13] Angular Velocity [wx, wy, wz] in world frame
    ], axis=-1)  # (T, 13)

    
    up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
    gravity_vec = get_axis_params(-1., up_axis_idx)

    # Convert to body frame (following Isaac Gym convention)
    base_quat = root_states[:, 3:7]  # (T, 4)
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])  # (T, 3) - now in body frame
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])  # (T, 3) - now in body frame
    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)  # (T, 3)
    #joint_pos = est["joint_positions"]       # (T, 12) 

    #Testing
    act_keys_pos = [f"{i:02d}_state_joint_position" for i in range(12)]
    joint_pos = np.stack([act[k] for k in act_keys_pos], axis=-1)  # (T, 12)

    joint_vel = est["joint_velocities"]      # (T, 12)
    cmd_lin = cmd["linear"]                  # (T, 3)
    cmd_ang = cmd["angular"]                 # (T, 3)
    # Convert commands to body frame 
    cmd_lin_body = quat_rotate_inverse(base_quat, cmd_lin)
    cmd_ang_body = quat_rotate_inverse(base_quat, cmd_ang)
    
    """
    From Grand Tour Github: 
    Joint Naming 0-11: ['LF_HAA', 'LF_HFE', 'LF_KFE', 'RF_HAA', 'RF_HFE', 'RF_KFE', 'LH_HAA', 'LH_HFE', 'LH_KFE', 'RH_HAA', 'RH_HFE', 'RH_KFE']
    """
    act_keys = [f"{i:02d}_command_position" for i in range(12)]

    actions = np.stack([act[k] for k in act_keys], axis=-1)   # (T, 12)
    if cfg.scale_actions:
        actions = make_actions_compatible(actions)
    else:
        actions = actions

    prev_actions = np.zeros_like(actions)
    prev_actions[1:] = actions[:-1]

    # select correct command dimensions to match isaac
    # cmd_lin is [vx, vy, vz], need [vx, vy] 
    # cmd_ang is [roll, pitch, yaw]. need [yaw]
    
    commands_xy_yaw = np.concatenate([
        cmd_lin_body[:, 0:2],   # Linear X and Y
        cmd_ang_body[:, 2:3]    # Angular Yaw
    ], axis=-1)      

    # re order concatenation to match Isaac Gym standard
    # BaseLin(3), BaseAng(3), Grav(3), Cmds(3), JointPos(12), JointVel(12), PrevAct(12)
    obs = np.concatenate([
        scale_lin_vel(base_lin_vel) if cfg.scale_lin_vel else base_lin_vel, # 3
        scale_ang_vel(base_ang_vel) if cfg.scale_ang_vel else base_ang_vel, # 3
        projected_gravity,  # 3
        scale_commands(commands_xy_yaw) if cfg.scale_commands else commands_xy_yaw, # 3
        scale_joint_pos(joint_pos) if cfg.scale_joint_pos else joint_pos, # 12
        scale_joint_vel(joint_vel) if cfg.scale_joint_vel else joint_vel, # 12
        #prev_actions,       # 12
    ], axis=-1)             # Total: 48 dimensions

    """
    From anymal legged gym source code:

    self.obs_buf = torch.cat((
    self.base_lin_vel * self.obs_scales.lin_vel,        # 3 dims
    self.base_ang_vel * self.obs_scales.ang_vel,          # 3 dims
    self.projected_gravity,                               # 3 dims
    self.commands[:, :3] * self.commands_scale,           # 3 dims (lin_vel_x, lin_vel_y, ang_vel_yaw)
    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12 dims
    self.dof_vel * self.obs_scales.dof_vel,              # 12 dims
    self.actions                                         # 12 dims
    ), dim=-1)
    """
    # Add noise to observations 
    #obs = add_observation_noise(obs)
    #obs = clip_observations(obs)
    
    # Clip torques to match Isaac Gym's torque limits (80.0 N⋅m for all joints from URDF)
    torque_limits = 80.0  # N⋅m, from ANYmal D URDF effort limits
    torques = np.clip(est["joint_efforts"], -torque_limits, torque_limits)
    
    rews, episode_sums_total = compute_rewards_offline(
        base_ang_vel,
        base_lin_vel,
        prev_actions,
        actions,
        joint_vel,
        joint_pos,  # joint_positions from state estimator
        est["LF_FOOT_contact"],
        est["LH_FOOT_contact"],
        est["RF_FOOT_contact"],
        est["RH_FOOT_contact"],
        cmd_lin_body,
        cmd_ang_body,
        torques,  # Use clipped torques instead of est["joint_efforts"]
        odom["pose_pos"],  # pose_pos for base height reward
        len(obs),
        return_per_term=True
    )

    # Shift for next_observations 
    observations = obs[:-1]
    next_observations = obs[1:]
    actions = actions[:-1]
    rewards = rews[:-1]

    # Terminals every 20s (20s * 50hz = 1000 steps)
    T = len(observations)
    episode_len = int(episode_len_s * hz)
    terminals = np.zeros(T, dtype=bool)
    terminals[np.arange(episode_len - 1, T, episode_len)] = True


    # offline dataset 
    dataset = dict(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        terminals=terminals,
    )

    return dataset, episode_sums_total

def save_offline_dataset_hdf5(dataset, filename="offline_dataset.hdf5"):
    with h5py.File(filename, "w") as f:
        for key, arr in dataset.items():
            print(f"Saving {key}: {arr.shape} {arr.dtype}")
            f.create_dataset(
                name=key,
                data=arr,
                compression="gzip",
                compression_opts=4,   
                chunks=True           
            )
    print(f"Saved dataset to {filename}\n")

def episode_returns(rewards, terminals):
    episode_sums = []
    current_sum = 0.0

    for r, done in zip(rewards, terminals):
        current_sum += r
        if done:
            episode_sums.append(current_sum)
            current_sum = 0.0

    if not terminals[-1]:
        episode_sums.append(current_sum)

    return np.array(episode_sums)

