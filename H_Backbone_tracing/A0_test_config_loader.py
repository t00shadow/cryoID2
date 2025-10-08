from A0_config_loader import load_sample_paths, load_config_section

sample_toggle = "emd_34843"
sample_toggle = "emd_42791"
sample_toggle = "real_malaria/1"

config_file = "H_Backbone_tracing/A0_config.yaml"

paths_cfg = load_sample_paths(config_file, sample_toggle)

# print(paths["input_mrc"])
for name, path in paths_cfg.items():
    # print(type(path))
    print(f"{name}: {path}")



clustering_cfg = load_config_section(config_file, "clustering")
output_dir = clustering_cfg["output_dir"]
threshold = clustering_cfg["threshold"]
print(output_dir)
print(threshold)

clustering_cfg = load_config_section(config_file, "samples")
print(clustering_cfg["emd_15905"]["base_dir"])



# copied from A1_clustering.py to check
FILE_TOGGLE = "emd_34843"
FILE_TOGGLE = "emd_42791"
FILE_TOGGLE = "real_malaria/1"

paths_cfg = load_sample_paths("H_Backbone_tracing/A0_config.yaml", FILE_TOGGLE)
INPUT_MRC = paths_cfg["CA_mrc"]
MASK_HELIX_MRC = paths_cfg["mask_helix_mrc"]
MASK_SHEET_MRC = paths_cfg["mask_sheet_mrc"]
MASK_RING_MRC = paths_cfg["mask_ring_mrc"]
ORIGINAL_MAP_PATH = paths_cfg["original_map"]

print("\n", INPUT_MRC, "\n", MASK_HELIX_MRC, "\n", MASK_SHEET_MRC, "\n", MASK_RING_MRC, "\n", ORIGINAL_MAP_PATH)
assert(False)