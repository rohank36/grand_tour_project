from pathlib import Path
import warnings
import zarr
warnings.filterwarnings('ignore')
import numpy as np
from reward import compute_rewards_offline
import h5py
from isaac_compatibility import *

def _to_np(x):
    return np.asarray(x[:]) if hasattr(x, '__getitem__') and not isinstance(x, np.ndarray) else np.asarray(x)

def _search_zoh_indices(src_ts, tgt_ts):
    """
    Vectorized zero order hold: for each target time, pick the last src index with src_ts <= tgt
    in english: find the index of the last source timestamp that happened at or before that time.
    Returns idx array (int) with -1 where no src sample exists yet
    """
    idx = np.searchsorted(src_ts, tgt_ts, side='right') - 1
    return idx

def _resample_group_zoh(group, tgt_ts, ts_key="timestamp", skip_keys=("timestamp","sequence_id")):
    """
    Resample all fields in a Zarr group to tgt_ts using ZOH.
    """
    out = {}
    src_ts = _to_np(group[ts_key])

    # Assure ascending timestamps
    if not np.all(src_ts[:-1] <= src_ts[1:]):
        order = np.argsort(src_ts)
        src_ts = src_ts[order]
        # Reorder all fields to keep arrays aligned
        for key in group.keys():
            if key in skip_keys: 
                continue
            arr = _to_np(group[key])
            out[key] = arr[order]  # temp store; we’ll overwrite after computing indices
        reordered = True
    else:
        reordered = False

    idx = _search_zoh_indices(src_ts, tgt_ts)  # -1 if tgt time is before first src sample
    # For each tgt time stamp, find which source timestamp (from og sensor) was the most recent reading that happened <= the tgt time
    # --> so idx are the row of the original sensor data to use for each new aligned time step

    # Build a safe index for gather; we’ll mask invalids later
    safe_idx = idx.copy()
    safe_idx[safe_idx < 0] = 0
    safe_idx[safe_idx >= len(src_ts)] = len(src_ts) - 1

    for key in group.keys():
        if key in skip_keys: 
            continue

        arr = _to_np(group[key]) if not (reordered and key in out) else out[key]
        # Gather
        res = arr[safe_idx]
        # Mask times before the first source sample (the -1s from _search_zoh_indices) as NaN 
        if res.dtype.kind in ('f',):  # floating types: use NaN
            res[idx < 0] = np.nan
        else:
            # For non-floats (ints, bools), you can choose a sentinel; here we keep first value.
            pass
        out[key] = res

    # Always return the resampled timestamps too (the grid)
    out["timestamp_50hz"] = tgt_ts
    return out

def _overlap_window(mission_root, sensors, ts_key="timestamp"):
    """Compute overlapping [start, end] across sensors to avoid extrapolation beyond last sample."""
    starts = []
    ends = []
    for s in sensors:
        ts = _to_np(mission_root[s][ts_key])
        starts.append(ts[0])
        ends.append(ts[-1])
    return max(starts), min(ends)

def build_50hz_grid(t_start, t_end, DT, TARGET_HZ):
    # Inclusive start, inclusive end if it lands exactly; otherwise stops before end
    n = int(np.floor((t_end - t_start) * TARGET_HZ)) + 1
    return (t_start + np.arange(n) * DT).astype(np.float64)

# main entrypoint
def align_mission_to_50hz(mission_root, sensors, DT, TARGET_HZ, ts_key="timestamp"):
    """
    Returns:
      {
        "t": np.ndarray [T],  # 50 Hz grid
        "sensors": {
            sensor_name: { field: np.ndarray[T, ...], "timestamp_50hz": np.ndarray[T] }
        }
      }
    """
    t0, t1 = _overlap_window(mission_root, sensors, ts_key=ts_key)
    tgt_ts = build_50hz_grid(t0, t1, DT, TARGET_HZ)

    aligned = {}
    for s in sensors:
        aligned[s] = _resample_group_zoh(mission_root[s], tgt_ts, ts_key=ts_key)

    return {"t": tgt_ts, "sensors": aligned}

    
dataset_folder = Path("~/Projects/rohan/grand_tour_project/grand_tour_code/missions").expanduser()
missions = [d.name for d in dataset_folder.iterdir() if d.is_dir()]
print(f"Total {len(missions)} missions")

topics = [
        "anymal_state_odometry",
        "anymal_state_state_estimator",
        "anymal_imu",
        "anymal_state_actuator",
        "anymal_command_twist",
        #"hdr_front",
        #"hdr_left",
        #"hdr_right"
]

TARGET_HZ = 50.0
DT = 1.0 / TARGET_HZ
aligned = {}

idx = 1
for mission in missions:

    print(f"Aligning {mission} ({idx}/{len(missions)})")

    mission_folder = dataset_folder / mission
    mission_root = zarr.open_group(store=mission_folder / "data", mode='r')

    if idx == 1:
        # init aligned data 
        aligned = align_mission_to_50hz(mission_root, topics, DT, TARGET_HZ)

    else:
        # build aligned data 
        aligned_tmp = align_mission_to_50hz(mission_root, topics, DT, TARGET_HZ)
        for k in list(aligned_tmp["sensors"].keys()):
            for k2 in list(aligned_tmp["sensors"][k].keys()):
                aligned["sensors"][k][k2] = np.concatenate((aligned["sensors"][k][k2], aligned_tmp["sensors"][k][k2]), axis=0)

    idx += 1

print("\nAlignment done.\n")

output_file = "aligned_dataset_shapes.txt"

with open(output_file, "w") as f:
    for sensor in topics:
        f.write(f"{sensor}:\n")
        keys = sorted(aligned["sensors"][sensor].keys())
        for key in keys:
            shape = aligned["sensors"][sensor][key].shape
            dtype = type(aligned["sensors"][sensor][key])
            f.write(f"-->    {key} {shape} {dtype}\n")
        f.write("------------------------------\n\n")

print(f"Sensor info written to {output_file}")

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

def build_offline_dataset(data, episode_len_s=20, hz=50):
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
    joint_pos = est["joint_positions"]       # (T, 12) # need to substract default_dos_pos here?
    joint_vel = est["joint_velocities"]      # (T, 12)
    cmd_lin = cmd["linear"]                  # (T, 3)
    cmd_ang = cmd["angular"]                 # (T, 3)

    base_lin_vel = scale_lin_vel(base_lin_vel)
    base_ang_vel = scale_ang_vel(base_ang_vel)
    joint_pos = scale_joint_pos(joint_pos)
    joint_vel = scale_joint_vel(joint_vel)
    
    """
    From Grand Tour Github: 
    Joint Naming 0-11: ['LF_HAA', 'LF_HFE', 'LF_KFE', 'RF_HAA', 'RF_HFE', 'RF_KFE', 'LH_HAA', 'LH_HFE', 'LH_KFE', 'RH_HAA', 'RH_HFE', 'RH_KFE']
    """
    act_keys = [f"{i:02d}_command_position" for i in range(12)]
    actions = np.stack([act[k] for k in act_keys], axis=-1)   # (T, 12)
    actions = make_actions_compatible(actions)

    prev_actions = np.zeros_like(actions)
    prev_actions[1:] = actions[:-1]

    # select correct command dimensions to match isaac
    # cmd_lin is [vx, vy, vz], need [vx, vy] 
    # cmd_ang is [roll, pitch, yaw]. need [yaw]
    
    commands_xy_yaw = np.concatenate([
        cmd_lin[:, 0:2],   # Linear X and Y
        cmd_ang[:, 2:3]    # Angular Yaw
    ], axis=-1)   
    commands_xy_yaw = scale_commands(commands_xy_yaw)         

    # re order concatenation to match Isaac Gym standard
    # BaseLin(3), BaseAng(3), Grav(3), Cmds(3), JointPos(12), JointVel(12), PrevAct(12)
    obs = np.concatenate([
        base_lin_vel,       # 3
        base_ang_vel,       # 3
        projected_gravity,  # 3
        commands_xy_yaw,    # 3  
        joint_pos,          # 12
        joint_vel,          # 12
        prev_actions,       # 12
    ], axis=-1)             # Total: 48 dimensions

    #TODO: if online eval not working correclty, try using actions instead of prev_actions to match Isaac exactly

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
    
    rews = compute_rewards_offline(
        base_ang_vel,
        base_lin_vel,
        prev_actions,
        actions,
        joint_vel,
        est["LF_FOOT_contact"],
        est["LH_FOOT_contact"],
        est["RF_FOOT_contact"],
        est["RH_FOOT_contact"],
        cmd_lin,
        cmd_ang,
        est["joint_efforts"],
        len(obs)
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

    return dataset

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

print("\nBuilding Offline Dataset...")
dataset = build_offline_dataset(aligned)
save_offline_dataset_hdf5(dataset)

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


for k in list(dataset.keys()):
    print(f"{k}: {type(dataset[k])} {dataset[k].shape}")

ep_ret = episode_returns(dataset["rewards"], dataset["terminals"])
print(f"Total episodes: {len(ep_ret)}")
print(f"Median Episode Return: {np.median(ep_ret)}")