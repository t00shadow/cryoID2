"""Shared CryoID orchestrator, lifecycle, and result contracts."""

from typing import Any

from .models import PipelineResult, StageResult

__version__ = "1.0.0"

__all__ = [
    "CryoIDPipeline",
    "PipelineCancelled",
    "PipelineResult",
    "StageResult",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Load the orchestrator lazily so lettered stages import independently."""

    if name in {"CryoIDPipeline", "PipelineCancelled"}:
        from .orchestrator import CryoIDPipeline, PipelineCancelled

        return {
            "CryoIDPipeline": CryoIDPipeline,
            "PipelineCancelled": PipelineCancelled,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
