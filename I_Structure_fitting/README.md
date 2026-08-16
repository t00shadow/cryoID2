# I - Structure retrieval and fitting

Production entry point: `I_Structure_fitting.stage`.

- `protein/` owns AlphaFold/public-PDB retrieval and protein raw-field fitting.
- `rna/` owns the validated public RNA-chain cache, candidate-to-structure
  lookup, RNA-chain isolation, and phosphate/scaffold fitting.

The orchestrator calls both APIs directly in Python. Structural scores select
fitted models but never rewrite the stage-D sequence ranking.
