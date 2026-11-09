# CryoID v2

Identify protein or RNA sequences and structures from cryo-EM density maps

## Requirements

- **Minimum OS and CPU architecture:** a **64-bit** system running **Linux x86-64
  with glibc >=2.17** (Conda installer), **Windows 10 version 1809+ or Windows 11
  x86-64**, or **macOS 14+ on Intel x86-64 or Apple Silicon arm64**.
  The Windows GUI minimum follows [Qt's platform requirements](https://doc.qt.io/qt-6.10/supported-platforms.html#windows).
- **Installer and Python:** [Miniforge](https://conda-forge.org/download/) **25.9.1-0+ (recommended)** or
  [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/)
  **25.5.1+**, initialized for your terminal. The installer creates **64-bit
  CPython 3.11** and installs Python/native dependencies automatically; CryoID's
  application supports CPython **3.10–3.13**.
- **GUI display:** a working desktop or remote graphical session. Linux needs
  an accessible X11 display (including XWayland or SSH X11 forwarding) and Qt's
  [X11/XCB, xkbcommon, font, and graphics runtime libraries](https://doc.qt.io/qt-6/linux-requirements.html).
  Conda supplies many of these libraries, but does not supply a display server.
  A terminal-only SSH session cannot display the GUI.
- **Native runtimes:** Linux needs the standard C/C++ and OpenMP runtimes
  (`libgcc`, `libstdc++`, `libgomp`) and compression libraries (`zlib`, `bzip2`).
  On **glibc 2.17–2.27**, the installer supplies a compatible BLAST build and its
  dependencies. On **glibc >=2.28**, the bundled BLAST executables also require
  **zlib >=1.2.9**. These bundled-BLAST requirements apply to pip-only Linux installs
  as well; see [BLAST runtime details](D_BLAST_searching/BUNDLED_BLAST.md#runtime-behavior).
  Windows needs the **x64 Microsoft Visual C++ v14 runtime**; Conda normally supplies
  it. If runtime DLLs are missing, install Microsoft's [supported x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist/).
- **Compute:** a GPU is optional; CPU execution is supported on all listed platforms.
  NVIDIA acceleration on Linux/Windows requires a supported GPU and compatible
  installed driver. CUDA 11.x minor compatibility starts at driver **450.80.02
  (Linux) / 452.39 (Windows)**, subject to successful execution checks; see
  [NVIDIA's compatibility limits](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
  Supported Blackwell GPUs need a CUDA 12.8-capable driver and, on Linux,
  **glibc >=2.28**. The installer selects and tests a build automatically, or uses
  CPU. A separately installed CUDA toolkit is unnecessary. On Macs, Auto uses
  Apple Metal (MPS) when the installed PyTorch and macOS can run the U-Net's
  operations; otherwise it uses CPU. An explicit MPS selection reports an error
  if the compatibility check fails. BLAST+ and other pipeline stages use CPU.
  For automatic GPU installation on a cluster, run setup in a GPU-allocated session.
- **Storage and permissions:** writable locations for the source checkout during
  setup, Conda environment/package cache, model cache, temporary files, and project
  outputs. Executables must be allowed to run from the installation and temporary
  directories. Published weights alone occupy about **3.03 GiB**; allow additional
  space for packages, downloads, structure caches, maps, and results.
- **Network and input:** HTTPS access for installation packages, first-use model
  downloads (unless local weights are supplied), and public structure retrieval.
  Provide a density map (`.mrc`, `.map`, `.ccp4`) and candidate FASTA.
  [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) is optional for viewing models
  and has its own system requirements.

The same Conda installation command below is used on every supported system.

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

## Protein complexes in the candidate FASTA

Use the same map and FASTA inputs in the CLI or GUI. To describe a complex,
give its chain records a shared `candidate` identifier and distinct `chain`
identifiers. Record identifiers must also be unique:

```fasta
>hemoglobin_alpha candidate=hemoglobin chain=alpha copies=2
VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR
>hemoglobin_beta candidate=hemoglobin chain=beta copies=2
VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH
```

`copies` defaults to 1; it specifies required assembly chain instances. A
single sequence with `copies=2` describes a homodimer. Independent candidates
and complexes can coexist in one FASTA. Records without `candidate=` remain
independent candidates.

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

## Running on HPC

Run the CLI on an allocated compute node using your site's scheduler (Slurm,
PBS, LSF, etc.). No X11 forwarding is needed.
Use the Conda installation above, in a GPU-allocated session for automatic GPU
detection. Use persistent project storage accessible from compute nodes, along
with accessible inputs/source; downloads need network access or pre-staged data.

Save this as `run-cryoid.sh`, replacing the paths:

```bash
#!/bin/bash
set -e
source /path/to/miniforge3/etc/profile.d/conda.sh
conda activate cryoid
python -m cryoid_pipeline.preflight
cryoid run --project /server/project --map /server/map.mrc \
  --pool /server/candidates.fasta --cpu-threads 4
```

Example submissions for [Slurm](https://slurm.schedmd.com/sbatch.html) or
[PBS Pro/OpenPBS](https://help.altair.com/2021.1.1/PBSProfessional/PBS2021.1.1.pdf).
The resource values are examples, not minimum requirements. Adapt account,
queue/partition, GPU binding, memory, and time to your site and workload:

```bash
# Slurm: one node, four CPU cores, one GPU
sbatch --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \
  --mem=32G --time=04:00:00 --output=cryoid-%j.log run-cryoid.sh

# PBS Pro/OpenPBS: same example resources; joined scheduler output
qsub -N cryoid -l select=1:ncpus=4:ngpus=1:mem=32gb \
  -l walltime=04:00:00 -j oe run-cryoid.sh
```
