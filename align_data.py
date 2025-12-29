from pathlib import Path
import warnings
import zarr
warnings.filterwarnings('ignore')
import numpy as np

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



def save_aligned_data(aligned, output_path="aligned_data.zarr"):
    """Save aligned sensor data to disk for later processing."""
    
    print(f"\nSaving aligned data to {output_path}...")
    root = zarr.open_group(store=output_path, mode='w')
    
    # Save timestamps
    root.create_dataset("t", data=aligned["t"], compression="gzip", compression_opts=4)
    
    # Save sensor data
    sensors_group = root.create_group("sensors")
    for sensor_name, sensor_data in aligned["sensors"].items():
        sensor_group = sensors_group.create_group(sensor_name)
        for field_name, field_data in sensor_data.items():
            print(f"  Saving {sensor_name}/{field_name}: {field_data.shape} {field_data.dtype}")
            sensor_group.create_dataset(
                field_name, 
                data=field_data, 
                compression="gzip", 
                compression_opts=4,
                chunks=True
            )
    
    print(f"Saved aligned data to {output_path}\n")

def load_aligned_data(input_path="aligned_data.zarr"):
    """Load aligned sensor data from disk."""
    
    print(f"Loading aligned data from {input_path}...")
    root = zarr.open_group(store=input_path, mode='r')
    
    aligned = {
        "t": np.array(root["t"]),
        "sensors": {}
    }
    
    for sensor_name in root["sensors"].keys():
        aligned["sensors"][sensor_name] = {}
        for field_name in root["sensors"][sensor_name].keys():
            aligned["sensors"][sensor_name][field_name] = np.array(
                root["sensors"][sensor_name][field_name]
            )
    
    print(f"Loaded aligned data from {input_path}\n")
    return aligned

def align_data(topics = None, save_aligned=False): 

    dataset_folder = Path("~/Projects/rohan/grand_tour_project/grand_tour_code/missions").expanduser()
    missions = [d.name for d in dataset_folder.iterdir() if d.is_dir()]
    print(f"Total {len(missions)} missions")

    if topics is None:
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
    if save_aligned:
        save_aligned_data(aligned, output_path="aligned_data.zarr")
    return aligned

if __name__ == "__main__":
    align_data(topics=None, save_aligned=True)