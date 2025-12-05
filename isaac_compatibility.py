import numpy as np

default_joint_angles = { # = target angles [rad] when action = 0.0
    "LF_HAA": 0.0,
    "LH_HAA": 0.0,
    "RF_HAA": -0.0,
    "RH_HAA": -0.0,

    "LF_HFE": 0.4,
    "LH_HFE": -0.4,
    "RF_HFE": 0.4,
    "RH_HFE": -0.4,

    "LF_KFE": -0.8,
    "LH_KFE": 0.8,
    "RF_KFE": -0.8,
    "RH_KFE": 0.8,
}

NUM_DOF = 12

# proper order as per anymal_d urdf 
DOF_NAMES = [
    "LF_HAA", 
    "LF_HFE", 
    "LF_KFE", 
    
    "RF_HAA", 
    "RF_HFE", 
    "RF_KFE", 

    "LH_HAA", 
    "LH_HFE", 
    "LH_KFE", 
    
    "RH_HAA", 
    "RH_HFE", 
    "RH_KFE"
    ]

action_scale = 0.5
obs_scale = 1.0

# Isaac Gym observation scaling factors
LIN_VEL_SCALE = 2.0
ANG_VEL_SCALE = 0.25

DOF_VEL_SCALE = 0.05  # From obs_scales.dof_vel
COMMANDS_SCALE = np.array([2.0, 2.0, 0.25])  # [lin_vel_x, lin_vel_y, ang_vel_yaw]

def build_default_dof_pos() -> np.ndarray:
    default_dof_pos = np.zeros(NUM_DOF, dtype=np.float32)
    for i in range(len(DOF_NAMES)):
        name = DOF_NAMES[i]  # Gets name in URDF order
        angle = default_joint_angles[name]  # Dictionary lookup by name
        default_dof_pos[i] = angle
    return default_dof_pos

def scale_joint_vel(joint_vel):
    return joint_vel * DOF_VEL_SCALE

def scale_commands(commands_xy_yaw):
    return commands_xy_yaw * COMMANDS_SCALE

def make_actions_compatible(absolute_positions):
    default_dof_pos = build_default_dof_pos()
    return (absolute_positions - default_dof_pos) / action_scale

def scale_lin_vel(lin_vel):
    return lin_vel * LIN_VEL_SCALE

def scale_ang_vel(ang_vel):
    return ang_vel * ANG_VEL_SCALE

def scale_joint_pos(joint_pos):
    default_dof_pos = build_default_dof_pos()
    return (joint_pos - default_dof_pos) * obs_scale
