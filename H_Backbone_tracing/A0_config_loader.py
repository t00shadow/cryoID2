from pathlib import Path
import yaml

def load_sample_paths(config_path="config.yaml", sample_toggle=None):
    """
    Load paths for a sample, automatically detecting files in the 'possibility' folder.
    The parameter sample_toggle controls which base_dir (see config file) is used. 'possibility' is relative to base_dir. {base_dir}/possibility
    Returns a dictionary with keys:
        - CA_mrc
        - mask_helix_mrc
        - mask_sheet_mrc
        - mask_ring_mrc
        - original_map
    If a file is missing, the value will be None.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if sample_toggle not in cfg["samples"]:
        raise ValueError(f"{sample_toggle} not found in config file")

    base_dir = Path(cfg["samples"][sample_toggle]["base_dir"])
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Base path is not a directory: {base_dir}")

    files_to_find = {
        "CA_mrc": base_dir / "possibility" / f"*_key_atoms_1_possibility.mrc",   # CA = c alpha
        "mask_helix_mrc": base_dir / "possibility" / f"*_secondary_strctures_1_possibility.mrc",
        "mask_sheet_mrc": base_dir / "possibility" / f"*_secondary_strctures_2_possibility.mrc",
        "mask_ring_mrc": base_dir / "possibility" / f"*_secondary_strctures_3_possibility.mrc",
        "original_map": base_dir / f"{sample_toggle}.map",    # added sample_toggle here just in case there's multiple map files in the base_dir
    }

    found_files = {}
    for key, pattern in files_to_find.items():
        # print(type(pattern))
        # Use glob to find the first file matching the wildcard in the possibility folder. There should NEVER be multiple matches. 
        matches = list(pattern.parent.glob(pattern.name))
        if len(matches) > 1:
            print(f"{len(matches)} wildcard match(es) for {pattern}")
            print(f"Too many matches. Check {pattern.parent} to resolve ambiguity.")    #! could consider raising an Exception here
        found_files[key] = matches[0] if matches else None

    return found_files

# Use this to load any other section
def load_config_section(config_path="config.yaml", section_name=None):
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if section_name not in cfg:
        raise KeyError(f"Section '{section_name}' not found in config file")

    return cfg[section_name]
