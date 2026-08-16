# I - Protein structure retrieval and fitting

The production API is `I_Structure_fitting.stage.run_protein_structure_pipeline`.
It constructs `core.pipeline.RunConfig` and calls `core.pipeline.run_csv`
directly. No repository script or subprocess wrapper sits between the
orchestrator and the Python implementation.

`core/` contains:

- candidate and density-input normalization;
- AlphaFold and public PDB retrieval;
- map placement utilities;
- rigid-body alignment and raw-field scoring;
- per-target summaries and fitted-model selection.

The production runner supports progress and cancellation callbacks supplied by
the orchestrator. External network services and ChimeraX remain optional
runtime integrations.

Paper-related Q-score and GT-only benchmarks are intentionally separate from
production at `experiments/structure_fitting/protein/`.
