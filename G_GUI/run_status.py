"""Whole-pipeline estimates driven by the same work progress as the GUI bar."""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import tempfile
import time
from pathlib import Path

from cryoid_pipeline.weights import pipeline_percent, DOWNLOAD_START, DOWNLOAD_END


# These are the production pipeline's progress boundaries. They describe work,
# not equal amounts of time: CPU inference and structure fitting differ greatly.
PHASES = (
    ("Input validation", 0, 3),
    ("U-Net prediction", 3, 55),
    ("Backbone tracing / query generation", 55, 57),
    ("Sequence search", 57, 59),
    ("Structure fetching / fitting", 59, 98),
    ("Finalizing outputs", 98, 100),
)
PHASE_INDEX = {name: index for index, (name, _start, _end) in enumerate(PHASES)}
PHASE_INDEX.update({"BLAST+ sequence search": 3, "RNA sequence ranking": 3})
MODELS = ("secondary_strctures", "key_atoms", "residue_types_6letter")


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


class RunStatus:
    def __init__(self, clock=time.monotonic, *, history_path: Path | None = None):
        self.clock = clock
        self.history_path = Path(history_path) if history_path is not None else None
        self.started = None
        self.ended = None
        self.stage = "Ready"
        self.device = "—"
        self.progress = "Waiting"
        self._percent = 0.0
        self._phase = 0
        self._phase_started = 0.0
        self._observed_at = 0.0
        self._fraction = 0.0
        self._budgets = [10.0, 1800.0, 60.0, 90.0, 600.0, 5.0]
        self._durations = [None] * len(PHASES)
        self._reused_predictions = False
        self._reused_phases = set()
        self._run_reused = False
        self._cancelling_since = None
        self._profile = ""
        self._download = None
        self._with_download = False

    def start(self, device: str, *, workload: dict | None = None) -> None:
        self.started = self.clock()
        self.ended = None
        self._percent = 0.0
        self._phase = 0
        self._phase_started = self.started
        self._observed_at = self.started
        self._fraction = 0.0
        self._durations = [None] * len(PHASES)
        self._reused_predictions = False
        self._reused_phases = set()
        self._run_reused = False
        self._cancelling_since = None
        self._download = None
        self._with_download = False
        workload = workload or {}
        # First-run budgets are deliberately provisional. Map size gives a
        # rough work scale; live measurements and matching history replace it.
        map_scale = max(0.125, min(64.0, float(workload.get("map_bytes", 8 * 1024**2)) / (8 * 1024**2)))
        self._budgets = [10.0, (180.0 if device.startswith("cuda") else 1800.0) * map_scale,
                         60.0 * map_scale, 90.0, 600.0, 5.0]
        self._profile = hashlib.sha256(json.dumps(
            {"device": device, "workload": workload}, sort_keys=True, default=str,
        ).encode()).hexdigest()
        records = self._read_history().get(self._profile, [])
        if isinstance(records, list):
            for index in range(len(PHASES)):
                values = [record[index] for record in records[-5:]
                          if isinstance(record, list) and len(record) == len(PHASES)
                          and self._valid_duration(record[index])]
                if values:
                    self._budgets[index] = statistics.median(values)
        self.set_stage({"name": "Starting", "device": device})

    def set_stage(self, event: dict) -> None:
        now = self.clock()
        kind = event.get("kind")
        if kind:
            # Detailed progress is independent of the throttled human log.
            # Keep the existing widgets/timing behavior without parsing the
            # new formatted console summaries.
            if kind == "stage_reused":
                phase = PHASE_INDEX.get(event.get("name"))
                if phase is not None:
                    self._reused_phases.add(phase)
            elif kind == "run_reused":
                self._run_reused = True
                self._reused_phases.update(range(len(PHASES)))
            elif kind == "inference_device":
                self.device = str(event["device"])
            elif kind == "inference_loading":
                self.progress = f"loading density map on {event['device']}"
            elif kind == "model_loading":
                self.progress = f"loading {event['model']} checkpoint {event.get('checkpoint', '')}"
            elif kind == "model_start":
                self.progress = f"running {event['model']}"
            elif kind == "model_progress":
                self._model_progress(str(event["model"]), int(event["completed"]), int(event["total"]))
            elif kind == "prediction_finished" and event.get("device") == "reused":
                self._reused_predictions = True
            return
        if event.get("name") == "Downloading weights":
            if self._download is None:
                self._download = {"started": now}
            self._with_download = True
            self._download.update(event, observed_at=now)
            completed, total = event.get("completed_bytes", 0), max(1, event.get("total_bytes", 1))
            self.stage, self.device = "Downloading weights", "Network"
            self.progress = f"{completed / 1024**2:.1f}/{total / 1024**2:.1f} MiB ({100 * completed / total:.1f}%)"
            self._percent = max(self._percent, DOWNLOAD_START + (DOWNLOAD_END - DOWNLOAD_START) * completed / total)
            return
        if self._download is not None and not str(event.get("name", "")).startswith("Cancelling"):
            # Download time is separate from validation/inference timing history.
            self._phase_started += now - self._download["started"]
            self._download = None
        self.stage = str(event.get("name", "Working"))
        self.device = str(event.get("device", "cpu"))
        self.progress = "Working…"
        if self.stage.startswith("Cancelling"):
            if self._cancelling_since is None:
                self._cancelling_since = now
            return
        index = PHASE_INDEX.get(self.stage)
        if index is not None and index > self._phase:
            self._record_phase(now)
            self._phase = index
            self._phase_started = now
            self._observed_at = now
            self._fraction = 0.0
            self.update_progress(pipeline_percent(PHASES[index][1], self._with_download))

    def update_progress(self, value: float) -> None:
        """Accept backend work progress; completion is confirmed by finish()."""
        if self.started is None or self.ended is not None or self._cancelling_since is not None:
            return
        if not math.isfinite(value) or value >= 100:
            return
        self._percent = max(self._percent, min(99.9, max(0.0, value)))
        if self._download is not None:
            return
        _name, lower, upper = PHASES[self._phase]
        lower = pipeline_percent(lower, self._with_download)
        upper = pipeline_percent(upper, self._with_download)
        fraction = min(1.0, max(0.0, (self._percent - lower) / (upper - lower)))
        if fraction > self._fraction:
            self._fraction = fraction
            self._observed_at = self.clock()

    @property
    def percent(self) -> int:
        return int(self._percent)

    def reset(self) -> None:
        """Clear the displayed run when selecting a different project."""
        self.started = self.ended = None
        self.stage = "Ready"
        self.progress = "Waiting"
        self._percent = 0.0
        self._cancelling_since = None
        self._download = None
        self._with_download = False

    def consume_log(self, message: str) -> None:
        if self.stage != "U-Net prediction":
            return
        if message.startswith("UNet inference: reusing verified project prediction maps"):
            self._reused_predictions = True
        if message.startswith("UNet inference: loading ") or message.startswith("UNet inference: running "):
            self.progress = message.removeprefix("UNet inference: ")
        match = re.fullmatch(r"UNet (\w+): predicted cube (\d+)/(\d+)", message)
        if match:
            model, completed, total = match.group(1), int(match.group(2)), int(match.group(3))
            self._model_progress(model, completed, total)

    def _model_progress(self, model: str, completed: int, total: int) -> None:
        self.progress = f"{model}: {completed}/{total} cubes ({round(100 * completed / max(total, 1))}%)"
        if model in MODELS and total > 0:
            index = MODELS.index(model)
            lower = 8 + round(47 * index / len(MODELS))
            upper = 8 + round(47 * (index + 1) / len(MODELS))
            self.update_progress(pipeline_percent(lower + (upper - lower) * min(completed / total, 1.0), self._with_download))

    def finish(self, label: str, *, reset_progress: bool = False) -> None:
        if self.started is None or self.ended is not None:
            return
        self.ended = self.clock()
        self.stage = label
        self.progress = "100%" if label == "Complete" else "Stopped"
        if label == "Complete" and not reset_progress:
            self._record_phase(self.ended)
            self._percent = 100.0
            if not self._run_reused:
                self._save_history()
        elif label == "Cancelled" or reset_progress:
            self._percent = 0.0

    def _record_phase(self, now: float) -> None:
        if self._phase not in self._reused_phases and not (self._phase == 1 and self._reused_predictions):
            self._durations[self._phase] = max(0.001, now - self._phase_started)

    @staticmethod
    def _valid_duration(value) -> bool:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and 0 < value < 365 * 86400 and math.isfinite(value))

    def _read_history(self) -> dict:
        if self.history_path is not None:
            try:
                data = json.loads(self.history_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("version") == 1 and isinstance(data.get("profiles"), dict):
                    return data["profiles"]
            except (OSError, ValueError):
                pass
        return {}

    def _save_history(self) -> None:
        if self.history_path is None:
            return
        temporary = None
        try:
            profiles = self._read_history()
            records = profiles.pop(self._profile, [])
            if not isinstance(records, list):
                records = []
            profiles[self._profile] = (records + [self._durations])[-5:]
            profiles = dict(list(profiles.items())[-32:])
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.history_path.parent,
                                             prefix=".timing-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump({"version": 1, "profiles": profiles}, handle)
            temporary.replace(self.history_path)
        except (OSError, ValueError):
            # An optional timing cache must never interrupt a scientific run.
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    @property
    def description(self) -> str:
        if self.progress in {"Waiting", "Working…", "Stopped", "100%"}:
            return self.stage
        detail = self.progress
        loading = re.match(r"loading (\w+) checkpoint ", detail)
        if loading:
            detail = f"Loading {loading.group(1)} weights"
        elif detail.startswith("loading density map on "):
            detail = "Loading density map"
        elif detail.startswith("running "):
            detail = f"Predicting {detail.removeprefix('running ')}"
        detail = (detail.replace("secondary_strctures", "secondary structures")
                  .replace("residue_types_6letter", "residue types")
                  .replace("key_atoms", "key atoms"))
        return f"{self.stage} · {detail}"

    @property
    def elapsed(self) -> str:
        return duration((self.ended if self.ended is not None else self.clock()) - self.started) if self.started is not None else "00:00:00"

    @property
    def remaining_seconds(self) -> float:
        if self.started is None or self.ended is not None:
            return 0.0
        now = self.clock()
        if self._cancelling_since is not None:
            age = now - self._cancelling_since
            return max(1.0, 5.0 - age, age * 0.1)
        if self._download is not None:
            remaining = self._download.get("remaining_seconds")
            if remaining is None:
                return float("inf")
            age = now - self._download["observed_at"]
            return max(1.0, remaining - age, 0.1 * age) + sum(self._budgets[1:])
        prior = self._budgets[self._phase]
        elapsed = now - self._phase_started
        if self._fraction > 0:
            # Five percent of prior work smooths early samples without a time
            # threshold. Even sub-second GPU cube updates improve the estimate.
            estimated_duration = (0.05 * prior + self._observed_at - self._phase_started) / (0.05 + self._fraction)
            projected = estimated_duration * (1.0 - self._fraction)
            age = now - self._observed_at
            current = max(1.0, projected - age, 0.05 * projected, 0.1 * age)
        else:
            # Stages without unit counts count down their historical budget.
            # If they overrun, expand the estimate instead of claiming zero.
            current = max(1.0, prior - elapsed, 0.05 * prior, 0.1 * elapsed)
        return current + sum(self._budgets[self._phase + 1:])

    @property
    def eta(self) -> str:
        remaining = self.remaining_seconds
        if not math.isfinite(remaining):
            return "Estimating download…"
        return f"~{duration(math.ceil(remaining))}" if remaining > 0 else "00:00:00"
