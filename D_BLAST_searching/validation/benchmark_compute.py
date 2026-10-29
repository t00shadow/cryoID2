"""Short, offline CPU benchmark using local checkpoints and existing map queries."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLBACKEND", "Agg")


def sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    import numpy as np
    import scipy.linalg
    import sklearn
    import torch
    from threadpoolctl import threadpool_info
    from cryoid_pipeline.compute import cpu_thread_limit, available_cpu_threads
    from cryoid_pipeline.user_config import load_config, resolve_weights_dir
    from C_Unet_model_predicting.unet_inference import MODEL_TYPES, normalize_map, predict_location, _load_checkpoint_data
    from C_Unet_model_predicting.model import UNET
    from D_BLAST_searching.blast_tools import bundled_blast_bin_dir
    from D_BLAST_searching.verify_bundle import ranking, rna_blast

    root = REPO / "D_BLAST_searching" / "validation"
    work = root / "work" / "compute_options"
    work.mkdir(parents=True, exist_ok=True)
    weights = resolve_weights_dir(None, load_config())
    if weights is None:
        raise ValueError("Configure the existing U-Net weights before benchmarking.")
    epochs = json.loads((weights / "epoch_num_path.txt").read_text())
    classes = json.loads((weights / "class_weight_for_training.txt").read_text())
    counts = [None, 4, 8]
    counts = [n for n in counts if n is None or n <= available_cpu_threads()]
    labels = ["auto" if n is None else str(n) for n in counts]
    report = {"platform": platform.platform(), "cpu": platform.processor(),
              "weights_dir": str(weights), "checkpoint_epochs": epochs,
              "available_cpu_threads": available_cpu_threads(), "cuda_available": torch.cuda.is_available(),
              "torch": torch.__version__, "torch_default_threads": torch.get_num_threads(),
              "default_pools": threadpool_info(), "repeats": args.repeats,
              "scope": "One real 48^3 map patch (one padded 64^3 cube), all three trained U-Net models; complete 4ZX8 ranking; cached RNA BLASTn lookup.",
              "inference": [], "ranking": {}, "rna_blastn": {}}
    def save_report():
        (root / "compute_options.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    density = REPO / "test_data/protein_e1d3/E1_D3_map.mrc"
    data, _zoom = normalize_map(density, recl=0.0)
    slices = tuple(slice(max(0, size // 2 - 24), max(0, size // 2 - 24) + 48) for size in data.shape)
    patch = np.zeros((48, 48, 48), dtype=np.float32)
    crop = data[slices]
    patch[tuple(slice(0, n) for n in crop.shape)] = crop
    inputs = {label: patch[None].copy() for label in labels}
    report["input_patch_sha256"] = sha(patch)
    for model_type in MODEL_TYPES:
        channels = len(classes[model_type])
        checkpoint = weights / "checkpoint" / model_type / f"{model_type}.epoch{epochs[model_type]}.pth.tar"
        print(f"Loading trained checkpoint: {model_type}", flush=True)
        model = UNET(inputs["auto"].shape[0], channels).to("cpu")
        checkpoint_data = _load_checkpoint_data(checkpoint, "cpu")
        model.load_state_dict(checkpoint_data.get("state_dict", checkpoint_data))
        del checkpoint_data
        gc.collect()
        predict_location(inputs["auto"], model, channels, "cpu")  # Untimed warm-up.
        baseline = None
        for count, label in zip(counts, labels):
            times = []
            with cpu_thread_limit(count):
                effective = torch.get_num_threads()
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    output = predict_location(inputs[label], model, channels, "cpu")
                    times.append(time.perf_counter() - started)
            if baseline is None:
                baseline = output.copy()
            item = {"model": model_type, "threads": label, "effective_torch_threads": effective,
                    "seconds": times, "median_seconds": statistics.median(times),
                    "probabilities_sha256": sha(output), "probabilities_bit_identical": bool(np.array_equal(output, baseline)),
                    "max_absolute_difference": float(np.max(np.abs(output - baseline))),
                    "different_argmax_voxels": int(np.count_nonzero(output.argmax(0) != baseline.argmax(0)))}
            report["inference"].append(item)
            inputs[label] = np.concatenate((inputs[label], output), axis=0)
            print(f"{model_type} {label}: {item['median_seconds']:.3f}s; different labels {item['different_argmax_voxels']}; max delta {item['max_absolute_difference']:.3g}", flush=True)
        del model, baseline, output
        gc.collect()
        save_report()
    cases = json.loads((root / "inputs.json").read_text())
    case = next(case for case in cases if case["name"] == "protein_4zx8")
    golden = json.loads((root / "windows-x86_64.json").read_text())["cases"][case["name"]]["bundled"]
    blast = bundled_blast_bin_dir()
    for count, label in zip(counts, labels):
        with cpu_thread_limit(count):
            result = ranking(case, work / "ranking" / label, None, cpu_threads=count)
            hits = rna_blast(work / "rna_blastn" / label, blast, cpu_threads=count)
        report["ranking"][label] = result
        report["rna_blastn"][label] = hits
        save_report()
        assert result["ranking_sha256"] == golden["ranking_sha256"], (label, "ranking scores differ")
        assert result["ranked_fasta_sha256"] == golden["ranked_fasta_sha256"], (label, "FASTA differs")
        assert hits["hits_sha256"] == report["rna_blastn"]["auto"]["hits_sha256"], (label, "BLASTn hits differ")
        print(f"Full ranking {label}: {result['seconds']:.3f}s; 166 candidates EXACT; RNA {hits['hit_rows']} hits EXACT", flush=True)
    report["restored_pools"] = threadpool_info()
    report["restored_torch_threads"] = torch.get_num_threads()
    assert report["restored_torch_threads"] == report["torch_default_threads"]
    report["rankings_exact"] = True
    report["argmax_labels_exact"] = all(item["different_argmax_voxels"] == 0 for item in report["inference"])
    save_report()
    print("Benchmark report saved to D_BLAST_searching/validation/compute_options.json", flush=True)


if __name__ == "__main__":
    main()
