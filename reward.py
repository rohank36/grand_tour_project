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
        base_height = -0. 
        feet_air_time =  1.0
        collision = -1.
        feet_stumble = -0.0 
        action_rate = -0.01
        stand_still = -0.

        hip_abduction_adduction = -0.0
        foot_drag = -0.0
        body_orientation = -0.0
    
    max_leg_spread = 0.5
    correct_sequence_reward = 0.5
    incorrect_sequence_penalty = -0.5
    only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
    tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
    soft_dof_vel_limit = 1.
    soft_torque_limit = 1.
    base_height_target = 1.
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

#------------ compute rewards  ----------------
def compute_rewards_offline(base_ang_vel,
                    base_lin_vel,
                    prev_actions,
                    actions,
                    joint_vel,
                    LF_contact, 
                    LH_contact,
                    RF_contact,
                    RH_contacnt,
                    linear_command,
                    angular_command,
                    torques,
                    T,
                    only_positive_rewards=True,
                    ):
    decimation = 4
    physics_base = 0.005
    dt = decimation * physics_base

    rew_buf = np.zeros(T, dtype=float)

    rew_buf += _reward_ang_vel_xy(base_ang_vel) * rewards.scales.ang_vel_xy * dt
    rew_buf += _reward_lin_vel_z(base_lin_vel) * rewards.scales.lin_vel_z * dt
    rew_buf += _reward_action_rate(prev_actions, actions) * rewards.scales.action_rate * dt
    rew_buf += _reward_dof_acc(joint_vel, dt) * rewards.scales.dof_acc * dt
    rew_buf += _reward_feet_air_time(LF_contact,LH_contact,RF_contact,RH_contacnt,linear_command,dt) * rewards.scales.feet_air_time * dt
    rew_buf += _reward_torques(torques) * rewards.scales.torques * dt
    rew_buf += _reward_tracking_ang_vel(angular_command, base_ang_vel) * rewards.scales.tracking_ang_vel * dt
    rew_buf += _reward_tracking_lin_vel(linear_command, base_lin_vel) * rewards.scales.tracking_lin_vel * dt

    if only_positive_rewards:
        #rew_buf[:] = np.clip(rew_buf[:], min=0.)
        rew_buf[:] = np.clip(rew_buf[:], a_min=0., a_max=None)

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

