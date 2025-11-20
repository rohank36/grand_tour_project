from huggingface_hub import HfApi,snapshot_download
from pathlib import Path
import os
import shutil
import tarfile
import re
import time
from dotenv import load_dotenv 

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") 

def move_dataset(cache, dataset_folder, allow_patterns=["*"]):
    print(f"Start moving from {cache} to {dataset_folder} !")

    def convert_glob_patterns_to_regex(glob_patterns):
        regex_parts = []
        for pat in glob_patterns:
            # Escape regex special characters except for * and ?
            pat = re.escape(pat)
            # Convert escaped glob wildcards to regex equivalents
            pat = pat.replace(r"\*", ".*").replace(r"\?", ".")
            # Make sure it matches full paths
            regex_parts.append(f".*{pat}$")

        # Join with |
        combined = "|".join(regex_parts)
        return re.compile(combined)

    pattern = convert_glob_patterns_to_regex(allow_patterns)
    files = [f for f in Path(cache).rglob("*") if pattern.match(str(f))]
    tar_files = [f for f in files if f.suffix == ".tar"]

    for source_path in tar_files:
        dest_path = dataset_folder / source_path.relative_to(cache)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with tarfile.open(source_path, "r") as tar:
                tar.extractall(path=dest_path.parent)
        except tarfile.ReadError as e:
            print(f"Error opening or extracting tar file '{source_path}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {source_path}: {e}")

    other_files = [f for f in files if not f.suffix == ".tar" and f.is_file()]
    for source_path in other_files:
        dest_path = dataset_folder / source_path.relative_to(cache)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

    print(f"Moved data from {cache} to {dataset_folder} !")


# Get all missions
repo_id = "leggedrobotics/grand_tour_dataset"
api = HfApi(token=HF_TOKEN)  
tree = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=False)

missions = [t.path for t in tree]
missions = missions[:-2] 
print(f"Total {len(missions)} missions")

# Define the destination directory
dataset_folder = Path("~/Projects/rohan/grand_tour_project/grand_tour_code/missions").expanduser()
dataset_folder.mkdir(parents=True, exist_ok=True)

counter = 0
for mission in missions:
    # You can change the mission and set the dataset_folder to your desired location.
    #mission = "2024-10-01-11-47-44"

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

    print(f"Downloading: {mission} ({counter}/{len(missions)})")

    allow_patterns = [f"{mission}/*.yaml", "*/.zgroup"]
    allow_patterns += [f"{mission}/*{topic}*" for topic in topics]
    hugging_face_data_cache_path = snapshot_download(
        repo_id="leggedrobotics/grand_tour_dataset", 
        allow_patterns=allow_patterns, 
        repo_type="dataset",
        resume_download=True,
        max_workers=2,       
        token=HF_TOKEN       
    )
    move_dataset(hugging_face_data_cache_path, dataset_folder, allow_patterns=allow_patterns)

    counter += 1

    time.sleep(10)

print("Done downloading.")