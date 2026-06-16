# Struct Fetch New Method

Standalone candidate discovery, rawfield rigid-body alignment, benchmark
Q-score evaluation, and plotting.

All implementation code lives under `src/`. The CLI files in `scripts/` are
thin shims that add `src/` to `sys.path` and call the corresponding module
entry point.

## Layout

```text
src/          pipeline, candidate fetching, rawfield alignment, Q-score, plotting
scripts/      CLI shims
configs/      editable JSON configuration files
test_data/    benchmark CSV plus optional local per-case assets
runs/         generated run directories
tmp/          on-demand root cache for the PDB chain pool FASTA
archive/      historical scripts, audits, and experiment scratch data
```

The root `tmp/` folder is only for the transient `pdb_seqres.txt.gz` candidate
pool. Run outputs are written under `runs/<run-name>/data/`.

## Pipeline

For each input CSV row, the production pipeline:

1. **Generates candidates** with `src/fetch.py`. The input sequence is aligned
   against every deduplicated protein chain sequence in the PDB pool. The
   candidate set is the union of PDB IDs whose chain sequence is tied for the
   top Smith-Waterman score.
2. **Adds AlphaFold candidates** when `include_alphafold=true`. UniProt
   accessions are derived from the candidate PDBs via PDBe mappings, then
   AlphaFold DB predictions are fetched for those accessions.
3. **Downloads PDB candidates** from RCSB.
4. **Aligns every candidate** to the cryo-EM map with `rawfield_ss_dominant`.
5. **Ranks candidates** by rawfield score, tracking the best AlphaFold model,
   best PDB model, and best overall model. With `keep_only_best_models=true`,
   losing source and fitted model files are deleted.

The CSV `gene_id` column is label-only. It is used for output filenames and log
lines, not for candidate retrieval. If it is blank, the pipeline falls back to
the top-hit UniProt accession as the label.

## Candidate Pool

- **Source**: RCSB `pdb_seqres.txt.gz`, one record per polymer chain.
- **Filter**: keep only `mol:protein` records.
- **Deduplication**: group by exact amino-acid sequence, with back-references
  to every `(PDB ID, chain ID)` that has that sequence.
- **Search**: parasail SIMD Smith-Waterman with BLOSUM62, `gap_open=11`, and
  `gap_extend=1`, matching BLAST blastp defaults.
- **Selection**: keep every unique sequence tied at the maximum SW score.
- **PDB candidates**: union of PDB IDs containing one of those tied top chains.
- **AlphaFold accessions**: UniProt IDs mapped from the candidate PDBs via
  PDBe `mappings/uniprot/{pdb}`.

The bundled benchmark CSV currently has 52 rows. Prior validation with this
workflow recovered the deposited GT PDB in the candidate list for all 52 cases,
but candidate sets can drift when RCSB updates `pdb_seqres.txt.gz`.

## Pool Lifecycle

The PDB chain pool is ephemeral per run:

- The first candidate-generation call downloads `pdb_seqres.txt.gz` from
  `https://files.rcsb.org/pub/pdb/derived_data/` into `tmp/`.
- Any stale on-disk pool file from a previous run is deleted before the fresh
  download.
- The pool is parsed once into memory and reused for subsequent candidate
  generation calls in the same process.
- `pipeline.run_csv` calls `fetch.cleanup_pool()` in a `finally` block, which
  clears the in-memory caches and deletes the on-disk pool file.

`benchmark_gt_only.py` never invokes candidate generation, so it never touches
the pool.

## Scoring Contract

- Alignment method: `rawfield_ss_dominant`
- Python implementation: `rawfield.run_raw_field`, called through
  `aligner.align_model_to_map`
- Rawfield runner id in metadata: `raw_field`
- Exported `alignment_runner` CSV/JSON label: `local_rawfield.run_raw_field`
- Ranking score: `raw_field_score`
- `score == raw_field_score`
- `score_metric == rawfield_ss_dominant_score`
- `final_map_meanz` and `map_cc_score` are diagnostics only

Q-score is computed separately by `qscore.py` through ChimeraX and never enters
candidate search or model ranking.

## Production Run

```powershell
python Alphafold_fetch\struct_fetch_new_method\scripts\run_struct_fetch_new_method.py
```

Key defaults live in `configs/pipeline.json`:

- `csv`: `test_data/ready_to_test_1_29.csv`
- `entries`: `1`; use `0` for all rows
- `run_dir`: empty string, which creates
  `runs/struct_fetch_rawfield_<timestamp>`
- `include_alphafold`: `true`
- `randomize_candidate_start`: `false`
- `keep_only_best_models`: `true`

The production runner performs candidate fetching and rawfield alignment only.
It does not compute Q-scores or generate plots.

Set `run_dir` in the config, or pass `--run-dir`, only when you intentionally
want to write into a fixed output directory.

### Input CSV

Required columns:

- `sequence`
- `density_dir`

Optional metadata columns:

- `gene_id`
- `recommended_contour_level`
- `ground_truth_model`
- `fitted_pdbs`
- `resolution`

Each `density_dir` must contain:

```text
ground_truth/<one .map/.mrc/.ccp4 file>
predictions/prediction_density_ca_key_atoms.mrc
predictions/prediction_density_secondary_structure_helix.mrc
predictions/prediction_density_secondary_structure_sheet.mrc
predictions/prediction_density_secondary_structure_loop.mrc
```

For benchmark rows with `ground_truth_model` or `fitted_pdbs`, the deposited GT
model is loaded from the sibling `models/` directory as
`ground_truth_model_<PDB>.cif`; `.mmcif` and `.pdb` are also accepted.

### Production Outputs

Production outputs are written under `runs/<run-name>/data/`:

```text
runs/<run-name>/data/sources/        downloaded AF/PDB candidate structures
runs/<run-name>/data/fitted_models/  aligned candidate structures
runs/<run-name>/data/logs/           run_config.json, summary.csv, summary.json, terminal.log, per-target candidate score CSVs
```

## Test Data

```text
test_data/ready_to_test_1_29.csv     active benchmark CSV, 52 rows
test_data/<case>/densities/          optional local per-case density directory
test_data/<case>/models/             optional local per-case deposited GT model
```

The large per-case `densities/` and `models/` folders are intentionally ignored
by git. Keep them locally, or point `density_dir` in your CSV at another
absolute or project-relative data location.

## Full Benchmark

The full benchmark runs the production pipeline, computes ChimeraX Q-scores for
the deposited GT model and selected fitted model, then generates the Q-score
scatter plot.

```powershell
python Alphafold_fetch\struct_fetch_new_method\scripts\benchmark_all.py
```

Key defaults live in `configs/benchmark_all.json`:

- `entries`: `0`, meaning all rows
- `run_dir`: empty string, which creates `runs/benchmark_all_<timestamp>`
- `include_alphafold`: `true`
- `run_pipeline`: `true`
- `compute_qscore`: `true`
- `generate_plot`: `true`
- `randomize_candidate_start`: `true`
- `random_start_margin_A`: `4.0`
- `keep_only_best_models`: `true`
- `qscore_workers`: `1`
- `qscore_timeout_sec`: `1800`

Set `CHIMERAX_PATH` or pass `--chimerax` if ChimeraX is not auto-detected.
Use `--skip-pipeline`, `--skip-qscore`, or `--skip-plot` for partial reruns.

Benchmark outputs are written under one timestamped run directory:

```text
runs/benchmark_all_<timestamp>/data/benchmark_config.json
runs/benchmark_all_<timestamp>/data/logs/terminal.log
runs/benchmark_all_<timestamp>/data/logs/summary.csv
runs/benchmark_all_<timestamp>/data/logs/summary.json
runs/benchmark_all_<timestamp>/data/qscore/benchmark_qscores.csv
runs/benchmark_all_<timestamp>/data/qscore/benchmark_qscores.json
runs/benchmark_all_<timestamp>/data/qscore/benchmark_qscores.png
```

Per-candidate rawfield scratch directories and per-Q-score-job temp directories
are removed after each finishes.

## GT-Only Benchmark

The GT-only benchmark skips candidate fetching. For each CSV row, it aligns the
deposited ground-truth model to the cryo-EM map, then computes Q-scores and
generates the alignment-only plot.

```powershell
python Alphafold_fetch\struct_fetch_new_method\scripts\benchmark_gt_only.py
```

Key defaults live in `configs/benchmark_gt_only.json`:

- `entries`: `0`, meaning all rows
- `run_dir`: `runs/gt_only_benchmark_current`
- `run_alignment`: `true`
- `compute_qscore`: `true`
- `generate_plot`: `true`
- `alignment_workers`: `4`
- `qscore_workers`: `2`
- `qscore_timeout_sec`: `1800`
- `randomize_candidate_start`: `true`
- `random_start_margin_A`: `4.0`
- `reuse_existing`: `true`
- `reuse_run_dirs`: empty by default
- `reuse_qscore_csvs`: empty by default

Because the default GT-only `run_dir` is fixed, set `run_dir` or pass
`--run-dir` when you want to keep multiple GT-only runs instead of reusing or
overwriting the current directory.

GT-only outputs include:

```text
runs/gt_only_benchmark_current/data/benchmark_gt_only_config.json
runs/gt_only_benchmark_current/data/logs/gt_only_summary.csv
runs/gt_only_benchmark_current/data/logs/gt_only_summary.json
runs/gt_only_benchmark_current/data/qscore/gt_only_qscores.csv
runs/gt_only_benchmark_current/data/qscore/gt_only_qscores.json
runs/gt_only_benchmark_current/data/qscore/gt_only_qscores.png
```

## Standalone Helpers

```powershell
python Alphafold_fetch\struct_fetch_new_method\scripts\qscore_model.py --model <model.cif> --map <map.map>
python Alphafold_fetch\struct_fetch_new_method\scripts\plot_qscore.py --csv <benchmark_qscores.csv> --out <plot.png>
```

## Archive

Historical setup, cleanup, audit, and experiment scripts live under `archive/`.
Nothing in `src/`, `scripts/`, or `configs/` references the archive; it is kept
only for provenance.
