# Compute options: short CPU benchmark

Tested on 2026-10-27 with the Intel Core Ultra 9 185H (16 cores, 22 logical
CPUs), Windows, and PyTorch 2.12.1+cpu. CUDA was unavailable, so this benchmark
does not measure GPU performance.

Auto applies no thread overrides: PyTorch kept 16 CPU threads, NumPy/SciPy
kept their 22-thread pools, and BLAST kept its single-thread search default.
Manual limits were applied to the supported numerical pools and BLAST; all
original pool limits were restored afterward, including on failure in tests.

## Real checkpoint inference

The input was a central 48×48×48 patch from the existing E1D3 density map,
normalized by the production code and processed as one padded 64×64×64 cube.
The three real trained checkpoints ran in their normal order, with each model's
probabilities feeding the next model. Checkpoint loading was excluded from
timing. Each model received one untimed Auto warm-up, then two timed passes per
thread setting. Values below are the median seconds per model.

| Model | Auto | 4 threads | 8 threads |
| --- | ---: | ---: | ---: |
| Secondary structures | 1.046 | 1.944 | 1.516 |
| Key atoms | 1.560 | 2.106 | 1.574 |
| Residue types | 1.306 | 2.183 | 1.631 |
| Sum of model medians | **3.911** | **6.234** | **4.721** |

All three settings produced **bit-identical probability arrays and identical
argmax labels** for every model in this sample. Auto was fastest for this
inference sample; its existing behavior remains the default.

## Ranking and lookup checks

- Complete protein ranking from the existing 4ZX8 map-derived queries:
  **166 candidates**, including scores and ordered FASTA, byte-identical at
  Auto, 4, and 8 threads and matching the saved pre-change reference.
- RNA structure lookup against the cached nucleotide database:
  **155 BLASTn hit rows**, byte-identical at all three settings.
- Regression suite: **152 tests and 19 subtests passed**, including GUI/CLI
  persistence, Auto behavior, manual limits, and restoring limits after errors.

This is a short check on representative inputs, not a full-map or end-to-end
speed benchmark. Ranking timings in the JSON include database/output work;
the first Auto call also includes cold plotting initialization, so those raw
times should not be used to compare thread performance. Matching these samples
does not guarantee identical floating-point results for every input or device.

Full metrics, pool limits, checkpoint epochs, and hashes are in
[`compute_options.json`](compute_options.json). To repeat from this source
checkout with the local datasets and configured weights:

```bash
python D_BLAST_searching/validation/benchmark_compute.py
```
