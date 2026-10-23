# CryoID v2

CryoID identifies protein or RNA sequences from cryo-EM density maps and fits
available public structures into the map.

## Requirements

- Windows 10/11 or 64-bit Linux; Python 3.10–3.13.
- A density map (`.mrc`, `.map`, or `.ccp4`) and candidate sequences in FASTA format.
- The external CryoID U-Net `train_val_data` weights bundle; **not included in Git**.
  Keep its checkpoint directories, `class_weight_for_training.txt`, and
  `epoch_num_path.txt` intact.
- NCBI BLAST+ (`blastp`, `blastn`, and `makeblastdb`) and internet access for
  public structure retrieval.

A CUDA-capable NVIDIA GPU with compatible PyTorch is recommended; CPU runs are
supported but slower. [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) is optional
and used for model visualization. Qt Designer is not required.

## Install

A fresh clone needs its dependencies and external weights before it can run.
From the repository folder:

**Linux:** with Conda or Miniforge installed:

```bash
conda env create -f environment.yml
conda activate cryoid
```

This installs CryoID, BLAST+, and the GUI runtime. For GPU use, verify that
PyTorch supports your NVIDIA driver.

**Windows:** in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[gui]"
```

Install [NCBI BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/)
separately on Windows. Activate the environment whenever you use CryoID.

## GUI workflow

Launch `cryoid-gui` in the installed environment, then:

1. **File → New:** create/select a project folder. Use **File → Open** to resume
   an existing project or run.
2. **Settings → Configurations:** set the weights directory and any undetected
   BLAST+/ChimeraX paths. **Results location** reflects your chosen project;
   change it here if needed.
3. Browse for the **Density Map** and **Candidate Pool**.
4. Adjust **Options** if needed: sequence mode, protein trace length, structure
   candidates, RCL, and temporary-file retention.
5. Press **Run**. It becomes **Abort** while running; progress and run messages
   appear in the progress bar and collapsible **Logs** box.
6. Under **Results**, **Sequence ranking** becomes available when a ranking
   exists, and **Model** when a fitted model exists. Model opens the map and
   fitted model in ChimeraX.

Without saved settings or overrides, a fresh source checkout uses
`<repository>/cryoid_results/`. Your chosen location is remembered. Selecting
an existing `cryoid_results` folder does not create a nested copy or move old
results. Theme switching is under **Settings → Theme**.

## Results and configuration

Results are stored in `<selected_folder>/cryoid_results/`:

- `runs/<run_id>/`: complete `sequence_ranking.tsv`, `alignment_summary.csv`,
  fitted models, `pipeline.log`, and `run_config.json`.
- `unetpredictions/`: prediction maps, reused only when provenance matches.
- `best_overall/`: best retained models across runs in this project.

The ranking viewer shows the first 20 rows; the saved TSV contains the full
candidate ranking. Structure selection defaults to one protein candidate or
50% of RNA candidates. An integer selects a count; a decimal selects a
fraction (`1` = one candidate, `1.0` = all). Temporary files are removed unless
retention is enabled in Options.

GUI and CLI share a per-user `config.toml`: `%APPDATA%/cryoid/` on Windows,
`~/.config/cryoid/` on Linux (or `$XDG_CONFIG_HOME/cryoid/`). `CRYOID_CONFIG`
can override its location.

## Command-line use

```bash
cryoid config set weights-dir /path/to/train_val_data
cryoid run /path/to/map.mrc /path/to/candidates.fasta --project-dir /path/to/output_parent
```

Use `cryoid config path` to locate the configuration and `cryoid run --help`
for run options. `--stop-after-identification` skips structure retrieval and
fitting. See [GUI details](G_GUI/README.md) for the desktop controls.
