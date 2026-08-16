# H - Backbone tracing and query generation

Production entry point: `H_Backbone_tracing.stage`.

- `common/`: shared prediction-volume clustering, graph, and volume primitives.
- `protein/`: helix masking, degree-limited
  tracing, six-class residue assignment, and protein query generation.
- `rna/`: paper-compatible phosphate/sugar scaffold tracing, two-letter
  residue assignment, and RNA query generation.

The effective mode selects one branch. Stage H receives only the prediction
channels applicable to that branch and returns a query FASTA bundle to the
orchestrator.

The cleaned RNA implementation was checked end to end against the archived
EMD-47655 prediction maps. It reproduced all seven paper-query FASTA records
exactly, with trace lengths 105, 96, 55, 44, 35, 23, and 14. The large maps
remain external and are not part of the source distribution.
