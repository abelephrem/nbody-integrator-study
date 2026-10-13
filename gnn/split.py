"""Train / val/ test split of the dataset, by whole file, within each setting combination."""

import glob
import json
import os

import h5py
import numpy as np

DATA_DIR = "C:/nbody_data"  # where the 360 HDF5 files live (set in one place only)
SPLIT_PATH = os.path.join(os.path.dirname(__file__), "splits", "split.json")  # gnn/splits/split.json
SEED = 0 

def read_tags(data_dir: str = DATA_DIR) -> list[dict]:
    """Read the metadata tags of every HDF5 file in data_dir.
    
    Returns one dictionary per file, in alphabetical order, e.g.
    {"file": "cluster_01000.h5", "split": "train", "type": "cluster",
    "N": 3, "mass_ratio": 1.0, "e": None}
    """
    tags = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):  # every .h5 file, alphabetical
        with h5py.File(path, "r") as f:
            a = f.attrs  # short name for the file's tags
            tags.append({
                "file": os.path.basename(path),  # filename only, no folder
                "split": str(a["split"]),  # "train" or a holdout tag like "N_extrap"
                "type": str(a["scenario_type"]),  # "two_body or "cluster"
                "N": int(a["N_bodies"]),
                "mass_ratio": float(a["mass_ratio"]),
                "e": float(a["e"]) if "e" in a else None,   # only two_body files have an eccentricity  
            })
    return tags


def setting_key(tag:dict) -> tuple:
    """The setting combination a file belongs to, used to group files."""

    if tag["type"] == "two_body":
        return ("two_body", tag["e"], tag["mass_ratio"])
    return ("cluster", tag["N"], tag["mass_ratio"])

def make_split(tags: list[dict], seed: int = SEED) -> dict:
    """Split the "train"-tagged files into train / val / test, by whole file."""

    # 1. sort files into groups by setting, holdout files set aside
    groups = {}  # setting tuple -> list of filenames
    holdout = []
    for tag in tags:
        if tag["split"] != "train":  # any holdout tag (N_extrap, e_interp ...)
            holdout.append(tag["file"])
            continue  # done with this file, go to the next
        key = setting_key(tag)
        if key not in groups:  # first file seen with this setting
            groups[key] = []
        groups[key].append(tag["file"])

    # 2. shuffle each group and deal it out
    rng = np.random.default_rng(seed)  #  random-number generator, reproducible from the seed
    split = {"train": [], "val": [], "test": []}
    for key in sorted(groups):  # sorted, so the groups are always visited in the same order
        files = sorted(groups[key])  # sorted, so the shuffle always starts from the same order
        order = rng.permutation(len(files))  # e.g. [2, 4, 3, 0, 1] for a group of 5
        files = [files[i] for i in order]  # the group's files in that shuffled order
        split["train"] += files[:-2]  # all but the last two -> train (3 of 5, or 8 of 10)
        split["val"].append(files[-2])  # second to last -> val
        split["test"].append(files[-1])  # last -> test
    split["holdout"] = holdout
    return split


def save_split(split: dict, seed: int = SEED, path: str = SPLIT_PATH) -> None:
    """Write the split ( and the seed that produced it) to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)  # create gnn/splits/ if it doesn't exist yet
    data = {
        "seed": seed,  # recorded so the split can be reproduced
        "train": split["train"],
        "val": split["val"],
        "test": split["test"],
        "holdout": split["holdout"],
    }
    with open(path, "w") as f:  # "w" = write (creates or replaces the file)
        json.dump(data, f, indent=2)  # indent=2: one entry per line, readable by eye


def load_split(path: str = SPLIT_PATH) -> dict:
    """Read the saved split back."""
    with open(path) as f:  # no "w" = open for reading
        return json.load(f)


if __name__ == "__main__":  # only runs with: python -m gnn.split
    split = make_split(read_tags(), SEED)
    save_split(split, SEED)
    print("saved to", SPLIT_PATH)
    print({name: len(files) for name, files in split.items()})
