from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


StageStatus = Literal["ok", "skipped", "blocked_missing_input", "error"]
RequestedSequenceMode = Literal["auto", "protein", "rna"]
SequenceMode = Literal["protein", "rna"]


@dataclass
class StageResult:
    name: str
    status: StageStatus
    message: str
    outputs: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionBundle:
    layout: str
    density_map: Path
    raw_prediction_dir: Path
    raw_key_atoms_1_path: Path | None
    secondary_structure_paths: list[Path]
    residue_type_paths: list[Path] = field(default_factory=list)
    density_dir: Path | None = None
    output_files: list[Path] = field(default_factory=list)
    mode: SequenceMode = "protein"
    key_atom_paths: list[Path] = field(default_factory=list)
    key_atom_channels: dict[int, Path] = field(default_factory=dict)
    secondary_structure_channels: dict[int, Path] = field(default_factory=dict)
    residue_type_channels: dict[int, Path] = field(default_factory=dict)

    def key_atom_path(self, channel: int) -> Path:
        if channel in self.key_atom_channels:
            return self.key_atom_channels[channel]
        if self.key_atom_paths:
            return self.key_atom_paths[channel - 1]
        if channel == 1 and self.raw_key_atoms_1_path is not None:
            return self.raw_key_atoms_1_path
        raise IndexError(f"Key-atom prediction channel {channel} is unavailable.")

    def secondary_structure_path(self, channel: int) -> Path:
        if channel in self.secondary_structure_channels:
            return self.secondary_structure_channels[channel]
        return self.secondary_structure_paths[channel - 1]

    def residue_type_path(self, channel: int) -> Path:
        if channel in self.residue_type_channels:
            return self.residue_type_channels[channel]
        return self.residue_type_paths[channel - 1]


@dataclass
class QueryBundle:
    query_fasta: Path
    source: str
    output_files: list[Path] = field(default_factory=list)


@dataclass
class SearchHit:
    query_id: str
    record_id: str
    score: float
    sequence: str
    description: str = ""
    chains: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineResult:
    run_dir: Path
    stages: list[StageResult]
    selected_hit: SearchHit | None = None
    output_files: list[Path] = field(default_factory=list)
    mode: SequenceMode | None = None
    ranking_path: Path | None = None

    @property
    def ok(self) -> bool:
        return any(stage.status == "ok" for stage in self.stages) and not any(
            stage.status == "error" for stage in self.stages
        )


def pipeline_result_to_dict(result: PipelineResult) -> dict[str, Any]:
    """Convert a pipeline result into the JSON-safe CLI wire format."""

    payload = {
        "run_dir": str(result.run_dir),
        "stages": [
            {
                "name": stage.name,
                "status": stage.status,
                "message": stage.message,
                "outputs": stage.outputs,
                "details": stage.details,
            }
            for stage in result.stages
        ],
        "selected_hit": None
        if result.selected_hit is None
        else {
            "query_id": result.selected_hit.query_id,
            "record_id": result.selected_hit.record_id,
            "score": result.selected_hit.score,
            "sequence": result.selected_hit.sequence,
            "description": result.selected_hit.description,
        },
        "output_files": [str(path) for path in result.output_files],
        "mode": result.mode,
        "ranking_path": str(result.ranking_path) if result.ranking_path is not None else "",
    }
    if result.selected_hit is not None and result.selected_hit.chains:
        payload["selected_hit"]["chains"] = result.selected_hit.chains
    return payload


def pipeline_result_from_dict(payload: dict[str, Any]) -> PipelineResult:
    """Reconstruct a pipeline result received from the command-line runner."""

    stages = [
        StageResult(
            name=str(item.get("name", "Stage")),
            status=str(item.get("status", "error")),
            message=str(item.get("message", "")),
            outputs=dict(item.get("outputs") or {}),
            details=dict(item.get("details") or {}),
        )
        for item in payload.get("stages", [])
    ]
    hit_payload = payload.get("selected_hit")
    selected_hit = None
    if isinstance(hit_payload, dict):
        selected_hit = SearchHit(
            query_id=str(hit_payload.get("query_id", "")),
            record_id=str(hit_payload.get("record_id", "")),
            score=float(hit_payload.get("score", 0.0)),
            sequence=str(hit_payload.get("sequence", "")),
            description=str(hit_payload.get("description", "")),
            chains=list(hit_payload.get("chains") or []),
        )
    return PipelineResult(
        run_dir=Path(str(payload.get("run_dir", "."))).expanduser().resolve(),
        stages=stages,
        selected_hit=selected_hit,
        output_files=[Path(str(path)).expanduser().resolve() for path in payload.get("output_files", [])],
        mode=(str(payload.get("mode")) if payload.get("mode") in {"protein", "rna"} else None),
        ranking_path=(
            Path(str(payload["ranking_path"])).expanduser().resolve()
            if payload.get("ranking_path")
            else None
        ),
    )
