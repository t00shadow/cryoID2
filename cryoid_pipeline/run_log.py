"""Plain-text run reporting shared by the CLI and the GUI's CLI subprocess.

Scientific events stay unthrottled. Only human-readable progress snapshots are
limited, so pipes, copied GUI logs, and terminals get the same readable text.
"""
from __future__ import annotations

from collections import deque
import math
import re
import time
from typing import Callable


MODEL_NAMES = {"secondary_strctures": "Secondary structure", "key_atoms": "Key atoms",
               "residue_types_6letter": "Residue types"}


def duration(seconds: float) -> str:
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


class RunLog:
    def __init__(self, write: Callable[[str], None], *, clock=time.monotonic, interval: float = 5.0):
        self.write, self.clock, self.interval = write, clock, interval
        self.started = self.clock()
        self.stage_started = self.started
        self.stage = ""
        self.overall = 0
        self.model = ""
        self.model_started = None
        self.samples = deque()
        self.last_snapshot = -math.inf
        self.last_count = None
        self.finished = False

    def progress(self, value: int) -> None:
        # Some backend exits publish 100 even when blocked/cancelled. Only a
        # successful result confirms completion for the human-readable output.
        self.overall = max(self.overall, min(99, max(0, int(value))))

    def message(self, message: str) -> None:
        """Forward diagnostics; suppress only messages replaced by events."""
        if message.startswith("Stage: ") or re.fullmatch(r"UNet \w+: predicted cube \d+/\d+", message):
            return
        if message.startswith(("UNet inference: loading ", "UNet inference: running ",
                               "UNet inference: completed on ",
                               "UNet inference: reusing verified project prediction maps")):
            return
        self.write(message)

    def event(self, event: dict) -> None:
        now = self.clock()
        kind = event.get("kind", "")
        if kind == "inference_device":
            device = str(event["device"])
            if device.startswith("cuda"):
                self.write(f"[GPU] {event.get('device_name', 'NVIDIA GPU')} ({device})")
            elif device == "mps":
                self.write(f"[GPU] {event.get('device_name', 'Apple GPU (Metal)')} (mps)")
            else:
                self.write("[CPU] U-Net inference is running on CPU.")
        elif kind == "inference_loading":
            self.write("U-Net: loading and preparing the density map.")
        elif kind == "model_loading":
            self.model = str(event["model"])
            self.model_started = None
            self.samples.clear()
            label = MODEL_NAMES.get(self.model, self.model)
            self.write(f"[Model {event['model_index']}/{event['model_total']}] {label} U-Net - loading checkpoint")
        elif kind == "model_start":
            self.model = str(event["model"])
            self.model_started = now
            self.samples = deque([(now, 0)])
            self.last_snapshot = -math.inf
            self.last_count = None
            self.write("Predicting cubes; progress and ETA below refer to this model.")
        elif kind == "model_progress":
            self._cubes(event, now)
        elif kind == "model_writing":
            self.write("Writing this model's prediction maps.")
        elif kind == "model_finished":
            label = MODEL_NAMES.get(str(event["model"]), str(event["model"]))
            elapsed = now - self.model_started if self.model_started is not None else 0
            self.write(f"[OK] {label} model completed. Model elapsed: {duration(elapsed)}")
        elif kind == "prediction_finished":
            device = str(event.get("device", "unknown"))
            if device == "reused":
                self.write("[OK] Reused verified cached prediction maps; U-Net inference was not run.")
            else:
                label = f"GPU: {device}" if device.startswith("cuda") or device == "mps" else "CPU" if device == "cpu" else device
                self.write(f"[OK] U-Net inference completed successfully ({label}).")
                self.write(f"{event.get('generated_file_count', 0)} prediction files generated.")
        elif event.get("stage_index") is not None:
            if self.stage:
                self.write(f"Stage elapsed: {duration(now - self.stage_started)} | Run elapsed: {duration(now - self.started)}")
            self.stage = str(event["name"])
            self.stage_started = now
            self.write(f"[Stage {event['stage_index']}/{event['stage_total']}] {self.stage}")
            if self.stage == "U-Net prediction":
                self.write(f"U-Net device selection: {event['device']}; checking predictions/cache before inference.")
            elif event.get("device") == "cpu":
                self.write("Device: CPU - expected for this stage.")
            self.write(f"Overall pipeline: ~{self.overall}%")

    def _cubes(self, event: dict, now: float) -> None:
        if self.model_started is None:
            return
        completed, total = int(event["completed"]), int(event["total"])
        if total <= 0 or not 0 <= completed <= total:
            return
        self.samples.append((now, completed))
        # Keep a recent window, with one earlier observation as its baseline.
        while len(self.samples) > 2 and self.samples[1][0] <= now - 20:
            self.samples.popleft()
        if (now - self.last_snapshot < self.interval and completed != total) or completed == self.last_count:
            return
        self.last_snapshot, self.last_count = now, completed
        elapsed = now - self.model_started
        sample_time, sample_count = self.samples[0]
        seconds, cubes = now - sample_time, completed - sample_count
        rate = cubes / seconds if seconds > 0 and cubes > 0 else 0.0
        enough = rate > 0 and (elapsed >= 1 and completed >= 2 or completed == total)
        speed = f"{rate:.2f} cubes/sec" if enough else "Calculating..."
        eta = "00:00:00" if completed == total else (
            f"~{duration(max(1, math.ceil((total - completed) / rate)))}" if enough else "Calculating...")
        self.write(f"Cubes (current model): {completed}/{total} ({100 * completed / total:.1f}%) | Overall pipeline: ~{self.overall}%")
        self.write(f"Speed: {speed} | Model elapsed: {duration(elapsed)} | Model ETA: {eta} | Run elapsed: {duration(now - self.started)}")

    def finish(self, status: str) -> None:
        if self.finished:
            return
        self.finished = True
        now = self.clock()
        if self.stage:
            self.write(f"Stage elapsed: {duration(now - self.stage_started)}")
        if status == "complete":
            self.write("Overall pipeline: 100%")
        self.write(f"[{status.upper()}] Run {status}. Total elapsed: {duration(now - self.started)}")
