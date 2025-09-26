from pathlib import Path
import yaml

# Has some logic for building the paths on the individual mrc files
def load_sample_paths(config_path="config.yaml", sample_toggle=None):
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
    emd_id = sample_toggle.split("_")[1]

    paths = {
        "CA_mrc": base_dir / "possibility" / f"emd_{emd_id}.map_key_atoms_1_possibility.mrc",   # CA = c alpha
        "mask_helix_mrc": base_dir / "possibility" / f"emd_{emd_id}.map_secondary_strctures_1_possibility.mrc",
        "mask_sheet_mrc": base_dir / "possibility" / f"emd_{emd_id}.map_secondary_strctures_2_possibility.mrc",
        "mask_ring_mrc": base_dir / "possibility" / f"emd_{emd_id}.map_secondary_strctures_3_possibility.mrc",
        "original_map": base_dir / f"emd_{emd_id}.map",
    }

    return paths

# Use this to load any other section
def load_config_section(config_path="config.yaml", section_name=None):
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if section_name not in cfg:
        raise KeyError(f"Section '{section_name}' not found in config file")

    return cfg[section_name]
