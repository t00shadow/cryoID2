# CryoID v2

Identify protein or RNA sequences and structures from cryo-EM density maps

## Requirements

- Linux/Windows x86-64, or macOS 14+ on Intel or Apple Silicon.
- For conda installations: [Miniforge](https://conda-forge.org/download/) **25.9.1-0+ (recommended)** or
  [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/)
  **25.5.1+**.
- A density map (`.mrc`, `.map`, `.ccp4`) and candidate FASTA. Published U-Net
  weights download automatically on first use (about 3 GiB); local weights are optional.
- Internet access for public structures. [ChimeraX](https://www.cgl.ucsf.edu/chimerax/)
  is optional for viewing models.

NOTE: Linux/Windows can use NVIDIA CUDA with compatible PyTorch; Macs use CPU.

## Install and launch

Run from the repository folder. Complete setup once, then use the run commands
each time you open a new terminal.

**Linux:** with Miniforge or Conda installed:

```bash
# One-time setup
conda activate base
python -m cryoid_pipeline.create_conda_env

# Run CryoID
conda activate cryoid
cryoid-gui
```

**Windows:** with Miniforge or Conda installed:

Run in PowerShell or Miniforge/Anaconda Prompt. To enable PowerShell, run
`conda init powershell` once in Miniforge/Anaconda Prompt, then reopen PowerShell.

```powershell
# One-time setup
conda activate base
python -m cryoid_pipeline.create_conda_env

# Run CryoID
conda activate cryoid
cryoid-gui
```

Alternatively, install Python 3.13 **64-bit (x64)** and use PowerShell:

```powershell
# One-time setup
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -e ".[gui]"
.\.venv\Scripts\python.exe -m cryoid_pipeline.preflight --gui

# Run CryoID
.\.venv\Scripts\Activate.ps1
cryoid-gui
```

**macOS:** with Miniforge or Conda installed:

```bash
# One-time setup
conda activate base
python -m cryoid_pipeline.create_conda_env

# Run CryoID
conda activate cryoid
cryoid-gui
```

### Installation validation and CLI-only installs

Use **64-bit CPython 3.10–3.13** on a supported architecture. The Conda setup
uses Python 3.11, installs native packages through Conda, and runs environment
validation after installation. On older Linux systems such as EICN, use this
route if compatible PyPI wheels are unavailable.

For a pip installation, `--only-binary=:all:` prevents dependencies from silently
falling back to source compilation, including pandas, PyQt6, and Pillow. If pip
reports no compatible distribution, use the Conda setup or a supported Python
and platform combination. See [pip's binary package options](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-only-binary).

To install **only the CLI**, without a Qt dependency:

```bash
python -m pip install --only-binary=:all: -e .
python -m cryoid_pipeline.preflight
```

`python -m pip install -e .` also installs only the CLI, but allows source builds
of dependencies. To add the GUI later, install `'.[gui]'` (PyQt6) or
`'.[gui-pyqt5]'` in the same environment. Afterward, run
`python -m cryoid_pipeline.preflight --gui`. Add `--device cuda:0` to test a CUDA build and GPU driver;
the headless check defaults to CPU and exits nonzero on failure.

The GUI checks setup at startup, after configuration changes, and again before
each run. It blocks inference on unsupported Python/architecture, missing or
incompatible declared dependency versions (including transitive requirements
such as Pillow), failed native imports, or failed PyTorch NumPy interchange and
3-D convolution on the selected device. BLAST validation verifies the bundled
files' SHA-256 checksums, requires BLAST+ 2.17.0 for all three executables, and
creates tiny databases to test protein and nucleotide searches. System BLAST
installations are not used. ChimeraX is optional and does not block inference.

These checks do not load the multi-GiB checkpoints: checkpoint/model
compatibility is checked when weights load. Small device checks also cannot
guarantee enough memory for every map. The headless check does not test display
server availability, even with `--gui`; the GUI needs a working desktop/display connection.

## Use (GUI)

1. **File → New:** enter a folder name and location in **New Project**, then click **Create**.
   **File → Open:** resume an existing project, its `cryoid_results` folder, or a run folder.
   The window title identifies the active project. Standard shortcuts are **Ctrl+N** / **Ctrl+O**
   on Windows/Linux and **⌘N** / **⌘O** on macOS. Project settings save automatically.
2. **Settings → Configurations:** optionally set **Custom weights path** (shared across projects).
   Leave it empty to autodownload published weights; the downloaded path is saved afterward.
   Optionally select ChimeraX (a `ChimeraX.app` path works on macOS).
3. Select the density map and candidate FASTA. Adjust **Options** if needed.
4. Click **Run**, then open **Sequence ranking** or **Model** when ready.

Use **Settings → Check setup** to diagnose missing requirements. **Abort** cancels
a run. Results are saved under `<project>/cryoid_results/`, with per-run outputs
in `runs/`, reusable predictions in `unetpredictions/`, and retained models in
`best_overall/`.

## Command line

Activate the environment first: `conda activate cryoid`, or
`.\.venv\Scripts\Activate.ps1` for the Windows venv method.

```bash
# First run (uses the current directory as the project)
cryoid run --map /path/to/map.mrc --pool /path/to/candidates.fasta

# Optional: use an existing local weights bundle instead of autodownloading
cryoid config set weights-dir /path/to/train_val_data

# Run an existing project with its saved settings
cryoid run --project /path/to/project

# View or change a project's saved settings
cryoid config --project /path/to/project
cryoid config --project /path/to/project set rcl 0.25
```

Add `--project /path/to/project` to `cryoid run` to choose another project.
`init` is optional. See `cryoid run --help` for options.
