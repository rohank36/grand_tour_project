import h5py
import numpy as np

input_path = "expert_dataset.hdf5"
output_path = "expert_dataset_fixed.hdf5"

with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
    for key in fin.keys():
        arr = fin[key][:]
        print(f"Original {key}: shape={arr.shape}, dtype={arr.dtype}")

        if key in ["rewards", "terminals"]:
            arr = np.asarray(arr).squeeze()
            if arr.ndim != 1:
                raise ValueError(
                    f"{key} is not 1D after squeeze, got shape {arr.shape}"
                )
            print(f"Fixed {key}: shape={arr.shape}, dtype={arr.dtype}")

        # Recreate dataset with same compression options
        fout.create_dataset(
            name=key,
            data=arr,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

print(f"\nWrote fixed dataset to {output_path}")
