from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_sample_dirs(root_dir):
    """
    Return all sample directories directly under root_dir.
    """
    root_dir = Path(root_dir)

    return sorted(
        d for d in root_dir.iterdir()
        if d.is_dir()
    )


def get_samples_with_results(root_dir):
    """
    Return sample folder names.
    """
    return [d.name for d in get_sample_dirs(root_dir)]


def get_file_per_sample(root_dir, filename, must_exist=True):
    """
    Return:
        {
            sample_id: Path(...)
        }

    Example:
        get_file_per_sample(outputs, "scores.csv")
    """
    files = {}

    for sample_dir in get_sample_dirs(root_dir):
        file_path = sample_dir / filename

        if file_path.exists():
            files[sample_dir.name] = file_path
        elif must_exist:
            raise FileNotFoundError(file_path)

    return files


def get_files_per_sample(root_dir, pattern):
    """
    Return:
        {
            sample_id: [Path(...), Path(...)]
        }

    Example:
        get_files_per_sample(outputs, "*.pdb")
    """
    files = {}

    for sample_dir in get_sample_dirs(root_dir):
        files[sample_dir.name] = sorted(sample_dir.glob(pattern))

    return files


# If frequently need a flat list across all samples
def get_all_files(root_dir, pattern):
    """
    Return a single flat list of matching files across all samples.
    """
    root_dir = Path(root_dir)

    files = []

    for sample_dir in get_sample_dirs(root_dir):
        files.extend(sorted(sample_dir.glob(pattern)))

    return files



# root = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\blast_results\outputs_batch_run_6_2_26\outputs"
root = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\blast_results\outputs_batch_run_6_15_26\outputs"
samples_with_results = get_samples_with_results(root)
# print(samples_with_results)
# print(len(samples_with_results))

score_files = get_file_per_sample(root, "queries_cutoff_length_10_optimized_method_summary.txt")
# print(score_files)
# print(len(score_files))

# txt_files = get_files_per_sample(root, "*.txt")
# print(txt_files[samples_with_results[0]])
# print(len(txt_files))

# all_files = get_all_files(root, "*.txt")
# print(all_files)
# print(len(all_files))


def parse_ave_identity(file_path):
    with open(file_path) as f:
        fields = f.readlines()[1].split()
    return float(fields[3].rstrip("%"))

def parse_composite_e(file_path):
    with open(file_path) as f:
        fields = f.readlines()[1].split()
    return float(fields[4])

# both in one fxn. just use this one?
def parse_result(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    fields = lines[1].split()

    return {
        "ave_identity": float(fields[3].rstrip("%")),
        "composite_E": float(fields[4]),
    }


# result = parse_ave_identity(score_files[samples_with_results[0]])
# result = parse_composite_e(score_files[samples_with_results[0]])
result = parse_result(score_files[samples_with_results[0]])
print(result)







METRIC = "ave_identity"  # "ave_identity" or "composite_E"

rows = []

sample_csv = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\New\ready_to_test_new.csv"
samples_df = pd.read_csv(sample_csv)

for _, sample in samples_df.iterrows():
    emdb_id = sample["emdb_id"].replace("-", "_")
    # print(emdb_id); assert False
    if emdb_id not in samples_with_results:
        print(f"{emdb_id} is missing results")
        continue

    resolution = sample["resolution"]

    result = parse_result(score_files[emdb_id])

    rows.append({
        "emdb_id": emdb_id,
        "resolution": resolution,
        **result,
    })

results_df = pd.DataFrame(rows)

# sort by resolution for a cleaner plot
results_df = results_df.sort_values("resolution")


### Calculating R^2
x = results_df["resolution"].to_numpy()
y = results_df[METRIC].to_numpy()
if METRIC == "composite_E":
    y = -np.log10(y)   # take negative log, so lower (better) E-value is higher on graph
# Linear regression
slope, intercept = np.polyfit(x, y, 1)
y_pred = slope * x + intercept
# R²
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"R² = {r2:.4f}")
print(f"{METRIC} = {slope:.4f} * resolution + {intercept:.4f}")


### Plotting
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,  # editable text in Illustrator
    "ps.fonttype": 42,
})
plt.figure(figsize=(4.5, 3.5), dpi=300)
plt.scatter(
    x,
    y,
    s=30,
    facecolors="white",
    edgecolors="black",
    linewidths=0.8,
)
# Regression line
plt.plot(
    x,
    y_pred,
    color="black",
    linewidth=2,
    label=f"R²={r2:.3f}"
)
plt.xlabel("Resolution (Å)")
plt.ylabel(METRIC)
plt.title(f"{METRIC} vs Resolution")
if METRIC == "composite_E":
    plt.ylabel(f"-log({METRIC})")
    plt.title(f"-log({METRIC}) vs Resolution")
plt.title("BLASTp Evaluation of Queries")
plt.legend(
    frameon=False,
    fontsize=10,
)
# plt.grid(
#     True,
#     color="0.9",
#     linewidth=0.5,
# )
# Remove top/right borders
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Slightly thicker remaining borders
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

plt.tight_layout()
plt.show()
