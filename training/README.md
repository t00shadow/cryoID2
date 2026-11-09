# U-Net training and dataset preparation

This directory provides recovered research code for inspecting and adapting the
cryoID2 U-Net training workflow. Routine protein and RNA identification uses
pretrained weights and does not require this directory or its environment.
Training source is available in repository checkouts but is excluded from the
pipeline wheel and source distribution. Training dependencies are separate from
the pipeline dependencies.

## Source and scope

The files under `source/` were recovered from `origin/feature/qibo`, commit
[`41fc3aab46e8bb3a1aa9ce9244bf4c4e364ca1a6`](https://github.com/t00shadow/cryoID_v2/tree/41fc3aab46e8bb3a1aa9ce9244bf4c4e364ca1a6),
on November 1, 2026. Their contents are preserved byte for byte.
[`source_manifest.json`](source_manifest.json) records each original path, Git
blob identifier, byte size, and SHA-256 hash. One dataset helper was relocated
from its original debug directory to satisfy the preparation script's import;
its contents were not changed. See [provenance](PROVENANCE.md).

| Location | Contents |
| --- | --- |
| `source/B_Unet_model_training/` | Training variants, the 3D U-Net, NumPy dataset loader, validation, checkpointing, and plotting utilities |
| `source/A_Dataset_generation/` | Historical download, normalization, labeling, cube generation, and splitting code |
| `source/U_UTILS/` | MRC helper used by dataset preparation, retaining its original attribution |
| `environment.yml` | Separate suggested compatibility environment |
| `label_groups_and_counts.txt` | Recovered submission record defining the full-to-reduced label mapping and voxel counts |
| `manuscript-notes.md` | Suggested Methods and Code availability wording |

The recovered code documents the implementation and several configurations. It
does not establish which exact revision and settings produced every released
checkpoint. Full training data, exact historical split manifests, the original
environment lockfile, and a complete checkpoint-to-run mapping are not included.
Full dataset generation and model training have not been rerun as part of this
source recovery.

## Environment

Use a separate environment, preferably on Linux with an NVIDIA GPU. The scripts
use CUDA automatic mixed precision; the pipeline's broader platform support
should not be interpreted as validation of these historical training scripts on
CPU, Windows, or macOS.

From the repository root:

```bash
conda env create -f training/environment.yml
conda activate cryoid-training
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The environment specifies Python 3.11, PyTorch 2.5.1, and NumPy 1.26.4 as a
compatibility starting point, not as the documented original training environment.
Install the appropriate PyTorch CUDA build for the training machine if needed.
PyTorch 2.5.1 retains the checkpoint-loading behavior used by the original code;
upgrading PyTorch can require adapting `torch.load` in a working copy. Optional
dataset downloading also requires the external `wget` and `gzip` commands.

## Configure a working copy

Copy `source/` to a separate working location before changing paths or settings.
Preserve this directory and its manifest as the source record. For example, on
Linux:

```bash
cp -R training/source /path/to/cryoid-training-run
cd /path/to/cryoid-training-run/B_Unet_model_training
```

The most relevant starting point for the six-letter residue model is
`A0_Unet_model_training-07092025-11192025-6letter.py`. It also supports selecting
the structure-class and key-atom models using `MODEL_INDEX`.

Edit the constants in the working copy before executing it:

| Setting | Meaning |
| --- | --- |
| `MAIN_PATH` | Dataset parent directory; the script reads `MAIN_PATH/train_val_data` |
| `MODEL_INDEX` | `0`: `secondary_strctures`; `1`: `key_atoms`; `2`: `residue_types_6letter` |
| `TRAIN`, `PREDICT`, `BATCH_1_TEST` | Select training, generation of intermediate probability arrays, or the single-batch diagnostic |
| `batch_size`, `num_workers` | Adjust to available memory and the training machine |
| `training_epoch_num` | Additional epochs attempted in that invocation, subject to early stopping |

The spelling `secondary_strctures` is retained because it is used in metadata,
folder names, and checkpoints. The selected script defines Adam with initial
learning rate `1e-4`, class-weighted cross-entropy, and `ReduceLROnPlateau` with
`mode='min'`, `factor=0.5`, and `patience=5`. It calls the scheduler with the
average validation loss and separately sets early-stopping patience to 15 epochs.
Other preserved variants have different settings; do not transfer these values
to all historical runs without checking the corresponding run records.

## Required inputs and training order

Supply `train_val_data/class_weight_for_training.txt` as JSON mapping model names
to their class-weight arrays. The six-letter workflow requires entries for
`secondary_strctures`, `key_atoms`, and `residue_types_6letter`. Output channel
counts, including the background class, are derived from these arrays; their
order must match the target labels.

Both `train_val_data/train/` and `train_val_data/val/` must contain:

| Folder | Arrays |
| --- | --- |
| `map_sample/` | Normalized density cubes, typically `(64, 64, 64)` |
| `secondary_strctures/` | Integer voxel labels for structure classes |
| `key_atoms/` | Integer voxel labels for key atoms |
| `residue_types_6letter/` | Integer voxel labels for the reduced residue/nucleotide classes |
| `secondary_strctures_pred_prob/` | Predicted structure-class probabilities, `(channels, 64, 64, 64)` |
| `key_atoms_pred_prob/` | Predicted key-atom probabilities, `(channels, 64, 64, 64)` |

The probability folders are generated between stages rather than supplied before
training the first model. The loader independently sorts filenames in each
folder and pairs arrays by their position. Ensure all required folders have
matching sample counts and consistent sample order, shapes, label encodings, and
dtypes.

1. Set `MODEL_INDEX = 0`, `TRAIN = True`, `PREDICT = False`, and
   `BATCH_1_TEST = False`; train the structure-class model.
2. Set `TRAIN = False`, `PREDICT = True` with the same model index and execute
   again to write its probability arrays for both training and validation data.
3. Repeat training and prediction with `MODEL_INDEX = 1` for the key-atom model.
4. Set `MODEL_INDEX = 2`, `TRAIN = True`, `PREDICT = False` to train the six-letter
   residue model using the map and both earlier probability arrays.

After each configuration change, run:

```bash
python A0_Unet_model_training-07092025-11192025-6letter.py
```

Checkpoints and evaluations are written under `train_val_data/checkpoint/` and
`train_val_data/evaluation/`. The script maintains `epoch_num_path.txt` to select
the saved epoch. Its resume path reloads model weights, but does not restore the
saved optimizer state, scheduler, scaler, or early-stopping history. It therefore
does not provide an exact continuation of all training state.

`A0_Unet_model_training.py` and the dated/CryoDataBot variants are preserved for
comparison, not as interchangeable entry points. In particular, the undated
script is saved with prediction enabled and training disabled and uses a
run-specific initial `best_loss` threshold. Review those settings before use.
The plotting utility also assumes the first validation batch has `batch_size`
samples; choose a suitable batch size or adapt that utility in the working copy.

## Dataset preparation

The recovered preparation entry point is
`source/A_Dataset_generation/B0_Dataset_generation.py`. In the working copy,
configure `MAIN_HOME_PATH`, the catalog path in `PATH_SETTINGS`, the output
`sample_path`, and the `DOWNLOAD`, `NORMALIZATION`, and `MAP2NPY` flags before
running:

```bash
cd /path/to/cryoid-training-run/A_Dataset_generation
python B0_Dataset_generation.py
```

The catalog must provide `emdb_id`, `fitted_pdbs`, and `resolution` columns in the
format expected by the script. The supplied helper covers structural labels,
key atoms, and full residue/nucleotide labels. It does **not** implement the
`residue_types_6letter` conversion. The exact intended mapping is preserved in
[`label_groups_and_counts.txt`](label_groups_and_counts.txt): background 0, six
protein groups 1-6, and RNA purine/pyrimidine groups 7-8. Correctly encoded reduced
labels and matching class weights must be supplied separately. The original
script applying that mapping to training cubes was not located in the repository
search described in [provenance](PROVENANCE.md#repository-search).

The historical entry point uses `splitfolders.ratio` with seed 44 and an 80/20
split of generated files. That is a code setting, not evidence of the split used
for the reported models or of separation by parent map or sequence family.
Define and preserve the appropriate split manifests before training new models.
The script moves generated files and removes its temporary staging directory;
use a dedicated output location when adapting it.

## Paper and release

See [manuscript notes](manuscript-notes.md) for draft availability wording.
Before citing this source as a published resource, make the documented directory
public and cite the exact manuscript release or archive DOI. A DOI has not been
created by adding this directory locally.
