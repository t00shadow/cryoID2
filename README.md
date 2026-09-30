# CryoID v2

CryoIDv2 identifies protein or RNA sequences from cryo-EM density maps and fits available public
structures into the map

## Requirements

- Windows 10/11 or 64-bit Linux
- Python 3.10-3.13
- A cryo-EM density map (`.mrc`, `.map`, or `.ccp4`)
- A protein or RNA candidate pool (`.fasta`, `.fa`, `.faa`, or `.fna`)
- The external CryoID U-Net `train_val_data` weights directory
- NCBI BLAST+ with `blastp`, `blastn`, and `makeblastdb`
- Internet access to retrieve the public structures used for fitting
- PyTorch with CUDA for normal production use; CPU execution is supported but
  slower

[UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/) is optional and is used
only to open results from the GUI.

The U-Net weights are not stored in this repository. The configured directory
must contain `class_weight_for_training.txt`, `epoch_num_path.txt`, and the
three checkpoint trees for `key_atoms`, `secondary_strctures`, and
`residue_types_6letter`. CryoID validates these files when a run starts.

## Installation

### Linux or server (recommended)

Use a current Miniforge or Conda installation and run these commands from the
repository root:

```bash
conda env create -f environment.yml
conda activate cryoid
```

The environment installs CryoID, its Python dependencies, BLAST+, and the GUI.
Legacy Conda 4.x may not understand current channel metadata. For GPU use,
make sure the installed PyTorch build is compatible with the server's CUDA
driver.

### Windows

Open PowerShell in the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[gui]"
```

Install BLAST+ from the
[official NCBI downloads](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/).

If PowerShell blocks environment activation, run this once in that terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Dependencies and entry points are defined in `pyproject.toml`.
`environment.yml` is the tested Conda setup

### Verify the installation

```bash
cryoid --help
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
blastp -version
blastn -version
makeblastdb -version
```

## Configuration

Launch the GUI and open **More Options > Advanced Settings**, or save the same
settings from a terminal:

```bash
cryoid config set weights-dir /path/to/train_val_data
cryoid config set project-dir /path/to/output_parent
cryoid config set blast-bin-dir /path/to/ncbi-blast/bin
cryoid config show
```

CryoID attempts to detect BLAST+ and the PyTorch device automatically. The GUI
and CLI share one TOML configuration file. Its location is shown by:

```bash
cryoid config path
```

| Setting | Meaning | Default |
| --- | --- | --- |
| `project-dir` | Parent in which `cryoid_results/` is created | current directory |
| `weights-dir` | U-Net `train_val_data` directory | required, unset |
| `blast-bin-dir` | Directory containing the three BLAST+ programs | autodetected |
| `device` | PyTorch device such as `cuda:0` or `cpu` | autodetected |
| `mode` | `auto`, `protein`, or `rna` | `auto` |
| `rcl` | Recommended contour level used for U-Net normalization | `0.000` |
| `protein-min-trace-length` | Minimum connected C-alpha nodes retained as a query | `10` |
| `structure-selection` | Leading ranked sequences sent to structure retrieval | protein: `1`; RNA: `0.5` (50%) |
| `keep-tmp` | Keep intermediate files and every tested model | `false` |
| `chimerax-path` | Optional ChimeraX executable | autodetected or unset |

`structure-selection` applies to the complete input-pool ranking before
structure lookup:

- `20` means the top 20 sequences.
- `0.2` means the top 20% of sequences.
- `1` means one sequence, while `1.0` means 100% of sequences.
- Non-integer values above 1 are invalid.

Fractions round up to the next sequence. A selected sequence without a public
structure still counts toward the selection.

## Running CryoID

Launch the desktop GUI:

```bash
cryoid-gui
```

Run from a terminal using saved settings:

```bash
cryoid run map.mrc candidate_pool.fasta
```

Override settings for one run when needed:

```bash
cryoid run map.mrc candidate_pool.fasta \
  --project-dir /path/to/output_parent \
  --weights-dir /path/to/train_val_data \
  --blast-bin-dir /path/to/ncbi-blast/bin \
  --mode auto --rcl 0 --structure-selection 1.0
```

Use `cryoid run --help` for all run options. Add
`--stop-after-identification` to generate predictions, queries, and the full
sequence ranking without retrieving or fitting structures.

## Pipeline behavior

1. The U-Net produces 15 probability maps.
2. Protein or RNA backbone tracing generates map-derived sequence queries.
3. Queries are used to rank input sequences.
4. The configured number or fraction of leading sequences is checked for
   public structures and the available models are fitted into the map.

`sequence_ranking.tsv` always contains every record from the original FASTA
pool in ranked order. Its first row is the top sequence hit. The GUI preview
shows only the first 20 rows; this does not change the saved ranking or the
structure selection.

Predictions are reused only when the density-map checksum, RCL, weights, and
all prediction checksums match their provenance JSON. Any mismatch regenerates
the complete 15-map bundle and replaces the old predictions after the new
inference succeeds.

## Outputs

The selected output path is always treated as a parent. CryoID creates or
reuses one `cryoid_results/` directory so files are not written directly into
the selected folder.

```text
<output_parent>/cryoid_results/
|-- best_overall/
|   |-- best_alphafold_model.<pdb|cif>          # protein, when available
|   |-- best_experimental_model.<pdb|cif>       # when available
|   `-- best_structure_selected_sequence.fasta
|-- unetpredictions/
|   |-- <map>_key_atoms_{1..3}_possibility.mrc
|   |-- <map>_secondary_strctures_{1..4}_possibility.mrc
|   |-- <map>_residue_types_6letter_{1..8}_possibility.mrc
|   `-- <map>_prediction_provenance.json
`-- runs/
    `-- <run_id>/
        |-- sequence_ranking.tsv
        |-- selected_structure_sequences.fasta
        |-- alignment_summary.csv
        |-- models/
        |   |-- alphafold/                       # protein only
        |   |   |-- fetched_models/
        |   |   `-- aligned_models/
        |   `-- experimental/
        |       |-- fetched_models/
        |       `-- aligned_models/
        |-- pipeline.log
        |-- run_config.json
        `-- tmp/                                 # only with Keep tmp
            |-- sequences_with_structures.tsv
            |-- unet/
            |-- tracing/
            |-- sequence_search/
            `-- structure_fetch/
```

- `best_overall/` contains the highest-scoring AlphaFold and experimental
  models across runs in that project. Its FASTA contains the sequence belonging
  to the higher-scoring retained model.
- `selected_structure_sequences.fasta` contains the sequences associated with
  the retained AlphaFold and experimental models. RNA normally has only the
  experimental record because the public AlphaFold Database provides protein
  predictions, not readily fetchable precomputed RNA models.
- `alignment_summary.csv` records every attempted model and fitting result,
  while the model directories retain only the best model of each type.
- `sequences_with_structures.tsv` is rebuilt from public structure data for
  each structure-fitting run. It and all other temporary data are deleted
  unless **Keep tmp** is enabled.
