"""Exercise timing with deterministic clocks; no inference or sleeping needed."""
import json
import math

import pytest

from G_GUI.run_status import RunStatus


STAGES = (
    "Input validation", "U-Net prediction", "Backbone tracing / query generation",
    "BLAST+ sequence search", "Structure fetching / fitting", "Finalizing outputs",
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def completed_run(path, durations=(2, 100, 10, 20, 200, 5), *, reuse=False, device="cpu", workload=None):
    clock = Clock()
    status = RunStatus(clock, history_path=path)
    status.start(device, workload=workload)
    for stage, seconds in zip(STAGES, durations):
        status.set_stage({"name": stage, "device": device if stage == "U-Net prediction" else "cpu"})
        if reuse and stage == "U-Net prediction":
            status.consume_log("UNet inference: reusing verified project prediction maps from cache")
        clock.advance(seconds)
        assert status.eta.startswith("~")
        assert math.isfinite(status.remaining_seconds) and status.remaining_seconds > 0
        assert status.percent < 100
    status.finish("Complete")
    assert status.percent == 100
    assert status.eta == "00:00:00"
    return status


def test_estimate_is_immediate_and_uses_device_and_map_size():
    clock = Clock()
    cpu, gpu, large = (RunStatus(clock) for _ in range(3))
    assert cpu.eta == "00:00:00"
    cpu.start("cpu")
    gpu.start("cuda:0")
    large.start("cuda:0", workload={"map_bytes": 32 * 1024**2})
    assert 0 < gpu.remaining_seconds < cpu.remaining_seconds
    assert gpu.remaining_seconds < large.remaining_seconds
    assert cpu.eta.startswith("~") and gpu.eta.startswith("~")


def test_same_progress_drives_whole_pipeline_eta_and_bar(tmp_path):
    history = tmp_path / "timing.json"
    completed_run(history)
    clock = Clock()
    status = RunStatus(clock, history_path=history)
    status.start("cpu")
    assert status.remaining_seconds == pytest.approx(337)
    clock.advance(2)
    status.set_stage({"name": "U-Net prediction", "device": "cpu"})
    clock.advance(50)
    status.update_progress(29)  # Half of the 3–55% inference range.
    assert status.percent == 29
    # 50 seconds of inference plus tracing/search/fitting/finalization.
    assert status.remaining_seconds == pytest.approx(285)
    clock.advance(25)
    status.update_progress(42)  # Three quarters of inference work.
    assert status.percent == 42
    assert status.remaining_seconds == pytest.approx(260)
    status.update_progress(20)  # A delayed/duplicate signal cannot rewind work.
    assert status.percent == 42
    assert status.remaining_seconds == pytest.approx(260)
    clock.advance(25)
    status.set_stage({"name": "Backbone tracing / query generation", "device": "cpu"})
    assert status.percent == 55
    assert status.remaining_seconds == pytest.approx(235)
    status.set_stage({"name": "Structure fetching / fitting", "device": "cpu"})
    clock.advance(100)
    status.update_progress(78.5)  # Half of the 59–98% fitting range.
    assert status.percent == 78
    assert status.remaining_seconds == pytest.approx(105)


def test_subsecond_gpu_samples_update_bar_and_eta_without_warmup():
    clock = Clock()
    status = RunStatus(clock)
    status.start("cuda:0")
    initial = status.remaining_seconds
    status.set_stage({"name": "U-Net prediction", "device": "cuda:0"})
    previous = status.percent
    for model in ("secondary_strctures", "key_atoms", "residue_types_6letter"):
        for index in range(1, 21):
            clock.advance(0.02)
            status.consume_log(f"UNet {model}: predicted cube {index}/20")
            assert status.eta.startswith("~") and "model" not in status.eta
            assert status.percent >= previous
            previous = status.percent
    assert clock.now < 10
    assert status.percent == 55
    assert 600 < status.remaining_seconds < initial  # Later CPU stages remain.


def test_estimate_survives_long_stalls_and_waits_for_confirmed_completion(tmp_path):
    history = tmp_path / "timing.json"
    completed_run(history)
    clock = Clock()
    status = RunStatus(clock, history_path=history)
    status.start("cpu")
    status.set_stage({"name": "Finalizing outputs", "device": "cpu"})
    clock.advance(5)
    short_wait = status.remaining_seconds
    clock.advance(100)
    assert status.remaining_seconds > short_wait
    assert status.eta.startswith("~")
    status.update_progress(100)
    assert status.percent == 98  # Backend can emit 100 even on error/cancel.
    status.finish("Complete")
    assert status.percent == 100
    assert status.eta == "00:00:00"


@pytest.mark.parametrize("outcome", ["Cancelled", "Failed", "Blocked"])
def test_stopped_runs_do_not_train_history_and_elapsed_freezes(tmp_path, outcome):
    history = tmp_path / "timing.json"
    completed_run(history)
    before = history.read_bytes()
    clock = Clock()
    status = RunStatus(clock, history_path=history)
    status.start("cpu")
    status.set_stage({"name": "U-Net prediction", "device": "cpu"})
    clock.advance(10)
    status.update_progress(20)
    if outcome == "Cancelled":
        status.set_stage({"name": "Cancelling / waiting for cleanup", "device": "cpu"})
        assert status.eta.startswith("~")
        status.update_progress(80)
        assert status.percent == 20
    status.finish(outcome)
    assert history.read_bytes() == before
    assert status.percent == (0 if outcome == "Cancelled" else 20)
    elapsed = status.elapsed
    clock.advance(1000)
    assert status.elapsed == elapsed
    assert status.eta == "00:00:00"
    status.start("cpu")
    assert status.percent == 0 and status.elapsed == "00:00:00"
    assert status.remaining_seconds == pytest.approx(337)


def test_prediction_cache_reuse_does_not_train_fresh_inference_time(tmp_path):
    history = tmp_path / "timing.json"
    completed_run(history)
    completed_run(history, (2, 0.01, 10, 20, 200, 5), reuse=True)
    status = RunStatus(Clock(), history_path=history)
    status.start("cpu")
    assert status.remaining_seconds == pytest.approx(337)


def test_history_is_separate_for_device_and_workload(tmp_path):
    history = tmp_path / "timing.json"
    completed_run(history)
    matching, gpu, other_map = (RunStatus(Clock(), history_path=history) for _ in range(3))
    matching.start("cpu")
    gpu.start("cuda:0")
    other_map.start("cpu", workload={"map_bytes": 64 * 1024**2})
    assert matching.remaining_seconds == pytest.approx(337)
    assert gpu.remaining_seconds != matching.remaining_seconds
    assert other_map.remaining_seconds > matching.remaining_seconds


@pytest.mark.parametrize("content", ["invalid JSON", '[]', '{"version": 99, "profiles": {}}'])
def test_bad_optional_history_does_not_prevent_estimation(tmp_path, content):
    history = tmp_path / "timing.json"
    history.write_text(content)
    status = RunStatus(Clock(), history_path=history)
    status.start("cpu")
    assert status.remaining_seconds > 0 and status.eta.startswith("~")


def test_malformed_history_records_and_unwritable_cache_are_harmless(tmp_path):
    history = tmp_path / "timing.json"
    completed_run(history)
    data = json.loads(history.read_text())
    key = next(iter(data["profiles"]))
    data["profiles"][key] = [None, {}, ["bad"] * 6, [float("nan")] * 6]
    history.write_text(json.dumps(data))
    status = RunStatus(Clock(), history_path=history)
    status.start("cpu")
    assert math.isfinite(status.remaining_seconds) and status.remaining_seconds > 0
    directory = tmp_path / "directory"
    directory.mkdir()
    completed_run(directory)  # Cannot replace a directory; timing stays optional.


def test_abort_with_cleanup_error_resets_progress_without_learning(tmp_path):
    history = tmp_path / "timing.json"
    clock = Clock()
    status = RunStatus(clock, history_path=history)
    status.start("cpu")
    status.update_progress(40)
    status.finish("Failed", reset_progress=True)
    assert status.percent == 0
    assert not history.exists()
    status.reset()
    assert status.stage == "Ready" and status.elapsed == status.eta == "00:00:00"
