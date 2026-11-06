"""Shared human output: deterministic timing, scope, devices, and throttling."""
import pytest

from cryoid_pipeline.run_log import RunLog


def make_log():
    now, lines = [0.0], []
    log = RunLog(lines.append, clock=lambda: now[0])
    return log, now, lines


def model_event(log, kind, **fields):
    log.event(dict(name="U-Net prediction", device="cpu", kind=kind,
                   model="secondary_strctures", model_index=1, model_total=3, **fields))


def test_cube_summary_is_throttled_and_excludes_checkpoint_loading():
    log, now, lines = make_log()
    model_event(log, "model_loading")
    now[0] = 30
    model_event(log, "model_start")
    model_event(log, "model_progress", completed=0, total=100)
    assert "Model ETA: Calculating..." in lines[-1]
    log.progress(18)
    for completed in range(1, 51):
        now[0] = 30 + completed / 10
        model_event(log, "model_progress", completed=completed, total=100)
    snapshots = [line for line in lines if line.startswith("Cubes")]
    assert len(snapshots) == 2  # Not fifty individual cube messages.
    assert snapshots[-1] == "Cubes (current model): 50/100 (50.0%) | Overall pipeline: ~18%"
    assert lines[-1] == ("Speed: 10.00 cubes/sec | Model elapsed: 00:00:05 | "
                         "Model ETA: ~00:00:05 | Run elapsed: 00:00:35")
    now[0] = 36
    model_event(log, "model_progress", completed=100, total=100)
    assert "100/100 (100.0%)" in lines[-2]  # Final snapshot is immediate.
    assert "Model ETA: 00:00:00" in lines[-1]


def test_each_model_resets_speed_and_eta_and_short_models_finish():
    log, now, lines = make_log()
    for index in range(2):
        model_event(log, "model_start")
        model_event(log, "model_progress", completed=0, total=1)
        assert "Calculating..." in lines[-1]
        now[0] += .25
        model_event(log, "model_progress", completed=1, total=1)
        assert "Speed: 4.00 cubes/sec" in lines[-1]
        model_event(log, "model_finished")
    assert sum("model completed" in line for line in lines) == 2


def test_recent_speed_does_not_average_over_entire_model():
    log, now, lines = make_log()
    model_event(log, "model_start")
    for second in range(31):
        now[0] = second
        model_event(log, "model_progress", completed=second, total=1000)
    for second in range(31, 56):
        now[0] = second
        model_event(log, "model_progress", completed=30 + 10 * (second - 30), total=1000)
    assert "Speed: 10.00 cubes/sec" in lines[-1]


@pytest.mark.parametrize("device,name", [("cuda:0", "NVIDIA GTX 1080 Ti"), ("mps", "Apple GPU (Metal)")])
def test_actual_device_and_expected_cpu_transition_are_explicit(device, name):
    log, now, lines = make_log()
    log.event(dict(name="U-Net prediction", device="auto", stage_index=2, stage_total=6))
    log.event(dict(kind="inference_device", device=device, device_name=name))
    log.event(dict(kind="prediction_finished", device=device, generated_file_count=30))
    now[0] = 12
    log.event(dict(name="Backbone tracing / query generation", device="cpu", stage_index=3, stage_total=6))
    assert f"[GPU] {name} ({device})" in lines
    assert f"[OK] U-Net inference completed successfully (GPU: {device})." in lines
    assert not any("running on CPU" in line for line in lines)
    assert "30 prediction files generated." in lines
    assert lines[-2] == "Device: CPU - expected for this stage."
    assert not any("fallback" in line.lower() or "failed" in line.lower() for line in lines)


def test_cached_predictions_do_not_claim_gpu_execution_or_generated_files():
    log, _, lines = make_log()
    log.event(dict(kind="prediction_finished", device="reused", generated_file_count=0))
    assert lines == ["[OK] Reused verified cached prediction maps; U-Net inference was not run."]


@pytest.mark.parametrize("status", ["complete", "blocked", "error", "cancelled"])
def test_only_success_reports_overall_completion(status):
    log, now, lines = make_log()
    log.progress(100)
    now[0] = 65
    log.finish(status)
    log.finish(status)
    assert ("Overall pipeline: 100%" in lines) == (status == "complete")
    assert lines[-1] == f"[{status.upper()}] Run {status}. Total elapsed: 00:01:05"
    assert sum("Total elapsed" in line for line in lines) == 1


def test_only_replaced_noise_is_suppressed_and_messages_are_plain_text():
    log, _, lines = make_log()
    for text in ("Stage: U-Net prediction; device: cpu", "UNet key_atoms: predicted cube 1/500",
                 "UNet inference: loading key_atoms checkpoint file.tar",
                 "UNet inference: running key_atoms", "UNet inference: completed on cpu; generated 30 file(s)"):
        log.message(text)
    assert lines == []
    important = ["Warning: CUDA failed; switching to CPU", "Checkpoint load error details",
                 "Run cancelled. Cleaning tmp/intermediate outputs.", "Could not download weights."]
    for text in important:
        log.message(text)
    assert lines == important
    assert not any("\r" in line or "\x1b" in line for line in lines)
