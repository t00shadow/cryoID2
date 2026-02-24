# Struct Fetch (basic)

This folder includes `struct_fetch_v4.py`, which:

- reads a CSV of protein sequences (and optional `gene_id`)
- fetches AlphaFold / PDB structure candidates
- fits candidates to an EM map
- writes fitted models + per-target score logs

## Quick start (simple)

1. Activate your Python environment for this repo.
2. Open `Alphafold_fetch/struct_fetch_v4.py` and edit the top settings:
   - `CSV_PATH` (your input CSV)
   - `MAP_PATH` (your `.mrc` map)
   - `ENTRIES_TO_PROCESS` (how many rows to run)
   - optional: `OUT_DIR` and `FITTED_DIR`
3. Make sure your CSV has a `sequence` column.
   - `gene_id` is optional (if empty, the script may use BLAST fallback when available).

Minimal CSV example:

```csv
gene_id,sequence
gdh,MHHHHH...YOUR_SEQUENCE...
,MSTNPKPQR...YOUR_SEQUENCE...
```

4. Run from the repo root (`cryoID_v2`):

```powershell
python Alphafold_fetch/struct_fetch_v4.py
```

## Output locations (default)

- Fetched source structures: `Alphafold_fetch/fetched_structures_v3/`
- Fitted models: `Alphafold_fetch/fitted_models_v4/`
- Score logs: `Alphafold_fetch/fitted_models_v4/logs/`

## Notes

- The script exits immediately if `CSV_PATH` or `MAP_PATH` does not exist.
- This script is configured by editing constants in the file (no command-line arguments yet).
- Internet access is required for UniProt / AlphaFold / PDB fetches.
