# Training source provenance

Source repository: https://github.com/t00shadow/cryoID_v2

Recovered reference: `origin/feature/qibo`, resolved to
`41fc3aab46e8bb3a1aa9ce9244bf4c4e364ca1a6` (December 9, 2025).

Recovery date: November 1, 2026.

All files in `source/` retain the exact bytes of the specified Git blobs. The
manifest records the source and destination paths and cryptographic hashes.
Compiled bytecode, notebooks, and backup directories were not copied. The root
repository license applies to the recovered project source; existing attribution
in the MRC helper has been retained.

The redundant `A0_Unet_model_training-07092025.py` variant was also omitted: its
original source contains a positional argument after a keyword argument in the
`UNET` constructor call, so it does not parse. The undated and six-letter scripts
retain the training implementation without altering that historical file.

## One relocated helper

The source commit contains `A_Dataset_generation/B0_Dataset_generation.py`, which
imports `B2_UTILS_data2npy`, but no longer contains that module at the expected
top-level location. An implementation remains at
`A_Dataset_generation/debug/B2_UTILS_data2npy.py`.

That file is included here as
`source/A_Dataset_generation/B2_UTILS_data2npy.py`, with identical content. Its
relative path setup resolves the recovered sibling `U_UTILS/MRC.py`. This
relocation supplies the import; it does not establish that this helper was the
one used for the final training dataset. Its full residue/nucleotide labels are
not the six-letter labels expected by the later residue-model training script.

## Relevant history

- `67caa1362e5fc8271146a59a180a485d0cb6ef69` (July 3, 2025) adds
  `ReduceLROnPlateau` and validation-loss-driven scheduler updates to
  `B_Unet_model_training/A0_Unet_model_training.py`.
- `41fc3aab46e8bb3a1aa9ce9244bf4c4e364ca1a6` contains the later training variants,
  including the six-letter residue-model script recovered here.
- `225374ef78637571f5d4ffa2517023ffd007d715` (August 16, 2026) removes the older
  training directory from the `combination` branch during pipeline replacement.

The README, environment specification, manifest, and manuscript notes are new
documentation for this recovery. The environment is a suggested compatibility
specification, not an original lockfile. Original script defaults have not been
silently changed to match manuscript statements or current software APIs.

## Repository search

A further search on November 1, 2026 examined local and remote branch references,
reflogs, deleted paths, historical notebook code, and unreferenced Git objects.
It searched 1,585 distinct project source/text revisions, checked 26 historical
dataset bytecode blobs for reduced-label references, and searched 9,248 stored
blobs smaller than 30 MB for six-letter naming, including blobs outside visible
branches. Other stored blobs were inventoried; large binary objects were not
treated as source code.

The training implementation, full-label generation, and downstream reduced
sequence conversion were found. The original code that converts full voxel
labels into `residue_types_6letter` training arrays was not located. This is a
search result, not a claim that the script never existed or cannot be recovered
from the original training computer or other records.

The exact grouping is available in `label_groups_and_counts.txt`, copied without
changes from the recovered submission record
`CryoID2/Other material for submission/Tables/U-Net/README.txt`. Its SHA-256 and
size are recorded separately from the Git sources in `source_manifest.json`.
This record documents background 0, protein groups 1-6, and RNA groups 7-8. It is
not executable preprocessing code. A newly implemented converter using that
mapping would need to be identified as a reconstruction rather than recovered
historical source.
