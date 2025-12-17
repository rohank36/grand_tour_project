import zarr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np

# from https://github.com/modanesh/legged_gym_anymal_d/blob/master/legged_gym/envs/base/legged_robot_config.py#L132

#------------ reward scales ----------------
class rewards:
    class scales:
        termination = -0.0
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        orientation = -0.
        torques = -0.00001
        dof_vel = -0.
        dof_acc = -2.5e-7
        base_height = -1.5
        feet_air_time =  1.0
        collision = -1.
        feet_stumble = -0.0 
        action_rate = -0.01
        stand_still = -0.

        hip_abduction_adduction = -1.0
        foot_drag = -0.027
        body_orientation = -0.0
    
    max_leg_spread = 0.5
    correct_sequence_reward = 0.5
    incorrect_sequence_penalty = -0.5
    only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
    tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
    soft_dof_vel_limit = 1.
    soft_torque_limit = 1.
    base_height_target = 0.515
    max_contact_force = 100. # forces above this value are penalized

#------------ reward functions----------------
        
def _reward_ang_vel_xy(base_ang_vel):
    # Penalize xy axes base angular velocity
    return np.sum(np.square(base_ang_vel[:, :2]), axis=1)

def _reward_lin_vel_z(base_lin_vel):
    # Penalize z axis base linear velocity
    return np.square(base_lin_vel[:, 2])

def _reward_action_rate(prev_actions, actions):
    # Penalize changes in actions
    return np.sum(np.square(prev_actions - actions), axis=1)

#def _reward_collision(self):
    #!!
    # Penalize collisions on selected bodies
    #return np.sum(1.*(np.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    # self.contact_forces comes from getting the contact force tensor from the sim. 
    # how should i replicate that for the offline dataset? 

def _reward_dof_acc(joint_vel, dt):
    # joint_vel = anymal_state_state_estimator["joint_velocities"]
    # Penalize dof accelerations

    # compute per-step acceleration (difference over time)
    dof_acc = (joint_vel[1:] - joint_vel[:-1]) / dt  # (T-1, num_joints)

    # squared L2 penalty per step
    reward_dof_acc = np.sum(np.square(dof_acc), axis=1)  # (T-1,)

    reward_dof_acc = np.concatenate([reward_dof_acc, [0.0]])

    return reward_dof_acc

def _reward_feet_air_time(LF,LH,RF,RH,linear_command,DT):
    #LF = anymal_state_state_estimator["LF_FOOT_contact"]
    #LH = anymal_state_state_estimator["LH_FOOT_contact"]
    #RF = anymal_state_state_estimator["RF_FOOT_contact"]
    #RH = anymal_state_state_estimator["RH_FOOT_contact"]
    # linear_command = anymal_command_twist["linear"]

    contacts = (np.stack([LF, LH, RF, RH], axis=1) > 0.5)
    T = contacts.shape[0]
    feet_air_time = np.zeros_like(contacts, dtype=float)  # (T,4)
    last_contacts = np.zeros_like(contacts, dtype=bool)   # (T,4)
    reward_air_time = np.zeros(T, dtype=float)
    moving = (np.linalg.norm(linear_command[:, :2], axis=1) > 0.1)

    for t in range(1, T):
        contact_filt = np.logical_or(contacts[t], last_contacts[t-1])
        first_contact = (feet_air_time[t-1] > 0.0) & contact_filt

        feet_air_time[t] = feet_air_time[t-1] + DT
        payout = np.maximum(feet_air_time[t] - 0.5, 0.0) * first_contact
        reward_air_time[t] = payout.sum()

        if not moving[t]:
            reward_air_time[t] = 0.0

        feet_air_time[t][contact_filt] = 0.0
        last_contacts[t] = contacts[t]

    return reward_air_time
    
    # Reward long steps
    # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    

def _reward_torques(torques):
    # torques = anymal_state_state_estimator["joint_efforts"]
    # Penalize torques
    return np.sum(np.square(torques), axis=1)

def _reward_tracking_ang_vel(angular_command, base_ang_vel):
    #base_ang_vel = anymal_state_state_estimator["twist_ang"]
    #commands_ang = anymal_command_twist["angular"]
    # Tracking of angular velocity commands (yaw) 
    ang_vel_error = np.square(angular_command[:, 2] - base_ang_vel[:, 2])
    return np.exp(-ang_vel_error/rewards.tracking_sigma)

def _reward_tracking_lin_vel(linear_command, base_lin_vel):
    #base_lin_vel = anymal_state_state_estimator["twist_lin"]      # (T, 3)
    #commands = anymal_command_twist["linear"]                     # (T, 3)
    # Tracking of linear velocity commands (xy axes)
    lin_vel_error = np.sum(np.square(linear_command[:, :2] - base_lin_vel[:, :2]), axis=1)
    return np.exp(-lin_vel_error/rewards.tracking_sigma)

def _reward_base_height(pose_pos):
    # pose_pos = anymal_state_state_estimator["pose_pos"] or anymal_state_odometry["pose_pos"]  # (T, 3)
    # Penalize base height away from target
    # Note: In Isaac Gym, this uses root_states[:, 2] - measured_heights, but we don't have terrain heights
    # So we use the absolute z position relative to target
    base_height = pose_pos[:, 2]  # z coordinate
    return np.square(base_height - rewards.base_height_target)

def _reward_hip_abduction_adduction(joint_pos):
    # joint_pos = anymal_state_state_estimator["joint_positions"]  # (T, 12)
    # Penalize wide spread between hip abduction/adduction joints
    hip_abduction_indices = [0, 3, 6, 9]  # Indices for LF_HAA, LH_HAA, RF_HAA, RH_HAA
    hip_positions = joint_pos[:, hip_abduction_indices]  # (T, 4)

    # Calculate the spread between front legs and between hind legs
    spread_front = np.abs(hip_positions[:, 0] - hip_positions[:, 2])  # LF - RF
    spread_hind = np.abs(hip_positions[:, 1] - hip_positions[:, 3])  # LH - RH

    # Penalize spreads exceeding a threshold
    threshold = rewards.max_leg_spread
    front_penalty = np.clip(spread_front - threshold, a_min=0, a_max=None)
    hind_penalty = np.clip(spread_hind - threshold, a_min=0, a_max=None)
    return front_penalty + hind_penalty

def _reward_foot_drag(LF_contact, LH_contact, RF_contact, RH_contact, joint_vel):
    # LF_contact, LH_contact, RF_contact, RH_contact = anymal_state_state_estimator["XX_FOOT_contact"]  # (T,)
    # joint_vel = anymal_state_state_estimator["joint_velocities"]  # (T, 12)
    # Penalize dragging motion of feet when in contact with the ground

    # Get contact information - use higher threshold to match original's selectivity
    # Original uses contact_forces > 1.0 N, so use higher threshold for normalized values
    contacts = np.stack([LF_contact, LH_contact, RF_contact, RH_contact], axis=1) > 0.8  # Changed from 0.5 to 0.8

    # Get foot velocities (approximating using the knee joint velocities)
    knee_indices = [2, 5, 8, 11]  # KFE joints (LF_KFE, RF_KFE, LH_KFE, RH_KFE)
    foot_velocities = joint_vel[:, knee_indices]  # (T, 4)

    # Penalize motion for feet in contact - SQUARE the velocities to match other penalties
    drag_penalty = np.where(contacts, np.square(foot_velocities), np.zeros_like(foot_velocities))

    # Sum penalties across all feet
    total_drag_penalty = np.sum(drag_penalty, axis=1)  # (T,)
    return total_drag_penalty


#------------ compute rewards  ----------------
def compute_rewards_offline(base_ang_vel,
                    base_lin_vel,
                    prev_actions,
                    actions,
                    joint_vel,
                    joint_pos,
                    LF_contact, 
                    LH_contact,
                    RF_contact,
                    RH_contact,
                    linear_command,
                    angular_command,
                    torques,
                    pose_pos,
                    T,
                    only_positive_rewards=True,
                    return_per_term=False,
                    ):
    decimation = 4
    physics_base = 0.005
    dt = decimation * physics_base

    rew_buf = np.zeros(T, dtype=float)
    episode_sums = {}  # Track per-term rewards for debugging

    # Compute each reward term separately
    rew_ang_vel_xy = _reward_ang_vel_xy(base_ang_vel) * rewards.scales.ang_vel_xy * dt
    rew_buf += rew_ang_vel_xy
    episode_sums['ang_vel_xy'] = np.sum(rew_ang_vel_xy)

    rew_lin_vel_z = _reward_lin_vel_z(base_lin_vel) * rewards.scales.lin_vel_z * dt
    rew_buf += rew_lin_vel_z
    episode_sums['lin_vel_z'] = np.sum(rew_lin_vel_z)

    rew_action_rate = _reward_action_rate(prev_actions, actions) * rewards.scales.action_rate * dt
    rew_buf += rew_action_rate
    episode_sums['action_rate'] = np.sum(rew_action_rate)

    rew_dof_acc = _reward_dof_acc(joint_vel, dt) * rewards.scales.dof_acc * dt
    rew_buf += rew_dof_acc
    episode_sums['dof_acc'] = np.sum(rew_dof_acc)

    rew_feet_air_time = _reward_feet_air_time(LF_contact,LH_contact,RF_contact,RH_contact,linear_command,dt) * rewards.scales.feet_air_time * dt
    rew_buf += rew_feet_air_time
    episode_sums['feet_air_time'] = np.sum(rew_feet_air_time)

    rew_torques = _reward_torques(torques) * rewards.scales.torques * dt
    rew_buf += rew_torques
    episode_sums['torques'] = np.sum(rew_torques)

    rew_tracking_ang_vel = _reward_tracking_ang_vel(angular_command, base_ang_vel) * rewards.scales.tracking_ang_vel * dt
    rew_buf += rew_tracking_ang_vel
    episode_sums['tracking_ang_vel'] = np.sum(rew_tracking_ang_vel)

    rew_tracking_lin_vel = _reward_tracking_lin_vel(linear_command, base_lin_vel) * rewards.scales.tracking_lin_vel * dt
    rew_buf += rew_tracking_lin_vel
    episode_sums['tracking_lin_vel'] = np.sum(rew_tracking_lin_vel)

    """
    rew_base_height = _reward_base_height(pose_pos) * rewards.scales.base_height * dt
    rew_buf += rew_base_height
    episode_sums['base_height'] = np.sum(rew_base_height)

    rew_hip_abduction_adduction = _reward_hip_abduction_adduction(joint_pos) * rewards.scales.hip_abduction_adduction * dt
    rew_buf += rew_hip_abduction_adduction
    episode_sums['hip_abduction_adduction'] = np.sum(rew_hip_abduction_adduction)

    rew_foot_drag = _reward_foot_drag(LF_contact, LH_contact, RF_contact, RH_contact, joint_vel) * rewards.scales.foot_drag * dt
    rew_buf += rew_foot_drag
    episode_sums['foot_drag'] = np.sum(rew_foot_drag)
    """

    if only_positive_rewards:
        #rew_buf[:] = np.clip(rew_buf[:], min=0.)
        rew_buf[:] = np.clip(rew_buf[:], a_min=0., a_max=None)

    if return_per_term:
        return rew_buf, episode_sums
    else:
        return rew_buf

        #for i in range(len(reward_functions)):
            #name = reward_names[i]
            #rew = reward_functions[i]() * reward_scales[name] * dt
            #rew_buf += rew
            #episode_sums[name] += rew
        
        # add termination reward after clipping only if episode ended early due to failure (e.g., robot falling)
        #if "termination" in reward_scales:
            #rew = _reward_termination() * reward_scales["termination"]
            #rew_buf += rew
            #episode_sums["termination"] += rew
        # ! don't need termination penalty because robot never fails 

