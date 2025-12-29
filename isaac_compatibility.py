import numpy as np

""" To make the GrandTour dataset data compatible with Isaac Gym """

# Isaac Gym default joint angles (target angles [rad] when action = 0.0)
isaac_default_joint_angles = {
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

# Grand Tour default joint angles (computed from mean of real robot data)
grand_tour_default_joint_angles = {
    "LF_HAA": -0.3818,
    "LF_HFE": 0.8445,
    "LF_KFE": -1.3450,
    "RF_HAA": 0.4191,
    "RF_HFE": 0.7747,
    "RF_KFE": -1.3316,
    "LH_HAA": -0.3890,
    "LH_HFE": -0.5935,
    "LH_KFE": 1.2910,
    "RH_HAA": 0.4031,
    "RH_HFE": -0.7371,
    "RH_KFE": 1.3671,
}

# Select which defaults to use for observation centering
# Set to 'grand_tour' to center observations using Grand Tour defaults
# Set to 'isaac' to use Isaac Gym defaults (original behavior)
OBS_CENTERING_MODE = 'grand_tour'

# Keep Isaac defaults for backward compatibility (used by make_actions_compatible)
default_joint_angles = isaac_default_joint_angles

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

# Clipping constants from legged_robot_config.py
CLIP_OBSERVATIONS = 100.0
CLIP_ACTIONS = 100.0

# Noise configuration from legged_robot_config.py
ADD_NOISE = True
NOISE_LEVEL = 1.0
NOISE_SCALES = {
    'lin_vel': 0.1,
    'ang_vel': 0.2,
    'gravity': 0.05,
    'dof_pos': 0.01,
    'dof_vel': 1.5,
}

def build_default_dof_pos(use_grand_tour=False) -> np.ndarray:
    """
    Build default DOF position array.
    
    Args:
        use_grand_tour: If True, use Grand Tour defaults. If False, use Isaac Gym defaults.
    
    Returns:
        np.ndarray of shape (12,) with default joint angles
    """
    angles_dict = grand_tour_default_joint_angles if use_grand_tour else isaac_default_joint_angles
    default_dof_pos = np.zeros(NUM_DOF, dtype=np.float32)
    for i in range(len(DOF_NAMES)):
        name = DOF_NAMES[i]  # Gets name in URDF order
        angle = angles_dict[name]  # Dictionary lookup by name
        default_dof_pos[i] = angle
    return default_dof_pos


def build_isaac_default_dof_pos() -> np.ndarray:
    """Build Isaac Gym default DOF positions (for action compatibility)."""
    return build_default_dof_pos(use_grand_tour=False)


def build_grand_tour_default_dof_pos() -> np.ndarray:
    """Build Grand Tour default DOF positions (for observation centering)."""
    return build_default_dof_pos(use_grand_tour=True)

def scale_joint_vel(joint_vel):
    return joint_vel * DOF_VEL_SCALE

def scale_commands(commands_xy_yaw):
    return commands_xy_yaw * COMMANDS_SCALE

def make_actions_compatible(absolute_positions):
    """
    Convert absolute joint positions to action space.
    
    Uses Grand Tour defaults so actions are centered around 0 for GT data.
    At inference, Isaac Gym must also use Grand Tour defaults.
    """
    default_dof_pos = build_grand_tour_default_dof_pos()
    return (absolute_positions - default_dof_pos) / action_scale

def scale_lin_vel(lin_vel):
    return lin_vel * LIN_VEL_SCALE

def scale_ang_vel(ang_vel):
    return ang_vel * ANG_VEL_SCALE

def scale_joint_pos(joint_pos):
    """
    Scale joint positions by centering around defaults.
    
    Uses Grand Tour defaults if OBS_CENTERING_MODE == 'grand_tour',
    otherwise uses Isaac Gym defaults.
    """
    use_grand_tour = (OBS_CENTERING_MODE == 'grand_tour')
    default_dof_pos = build_default_dof_pos(use_grand_tour=use_grand_tour)
    return (joint_pos - default_dof_pos) * obs_scale

def get_noise_scale_vec(obs_dim=48):
    """
    Compute noise scale vector for observations.
    Matches _get_noise_scale_vec from legged_robot.py
    For 48-dim observations (no height measurements):
    - [0:3]: lin_vel noise
    - [3:6]: ang_vel noise
    - [6:9]: gravity noise
    - [9:12]: commands (no noise)
    - [12:24]: dof_pos noise
    - [24:36]: dof_vel noise
    - [36:48]: previous actions (no noise)
    """
    noise_vec = np.zeros(obs_dim, dtype=np.float32)
    
    # lin_vel: noise_scales.lin_vel * noise_level * obs_scales.lin_vel
    noise_vec[0:3] = NOISE_SCALES['lin_vel'] * NOISE_LEVEL * LIN_VEL_SCALE
    
    # ang_vel: noise_scales.ang_vel * noise_level * obs_scales.ang_vel
    noise_vec[3:6] = NOISE_SCALES['ang_vel'] * NOISE_LEVEL * ANG_VEL_SCALE
    
    # gravity: noise_scales.gravity * noise_level
    noise_vec[6:9] = NOISE_SCALES['gravity'] * NOISE_LEVEL
    
    # commands: no noise (9:12)
    noise_vec[9:12] = 0.0
    
    # dof_pos: noise_scales.dof_pos * noise_level * obs_scales.dof_pos
    noise_vec[12:24] = NOISE_SCALES['dof_pos'] * NOISE_LEVEL * obs_scale
    
    # dof_vel: noise_scales.dof_vel * noise_level * obs_scales.dof_vel
    noise_vec[24:36] = NOISE_SCALES['dof_vel'] * NOISE_LEVEL * DOF_VEL_SCALE
    
    # previous actions: no noise (36:48)
    noise_vec[36:48] = 0.0
    
    return noise_vec

def add_observation_noise(observations, noise_scale_vec=None):
    """
    Add noise to observations matching Isaac Gym implementation.
    Noise is uniform in [-1, 1] scaled by noise_scale_vec.
    """
    if not ADD_NOISE:
        return observations
    
    if noise_scale_vec is None:
        noise_scale_vec = get_noise_scale_vec(observations.shape[-1])
    
    # Generate uniform noise in [-1, 1] and scale it
    # (2 * rand() - 1) gives uniform in [-1, 1]
    noise = (2 * np.random.rand(*observations.shape) - 1) * noise_scale_vec
    return observations + noise

def clip_observations(observations, clip_value=None):
    """
    Clip observations to [-clip_value, clip_value].
    Matches legged_robot.py line 101.
    """
    if clip_value is None:
        clip_value = CLIP_OBSERVATIONS
    return np.clip(observations, -clip_value, clip_value)

def clip_actions(actions, clip_value=None):
    """
    Clip actions to [-clip_value, clip_value].
    Matches legged_robot.py line 87.
    """
    if clip_value is None:
        clip_value = CLIP_ACTIONS
    return np.clip(actions, -clip_value, clip_value)
