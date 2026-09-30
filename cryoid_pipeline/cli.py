from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

from .models import PipelineResult, pipeline_result_from_dict, pipeline_result_to_dict
from .orchestrator import CryoIDPipeline
from .user_config import (
    default_config_path,
    load_config,
    resolve_device,
    resolve_keep_tmp,
    resolve_blast_bin_dir,
    resolve_project_dir,
    resolve_protein_min_trace_length,
    resolve_rcl,
    resolve_sequence_mode,
    resolve_structure_selection,
    resolve_weights_dir,
    update_config,
    validate_keep_tmp,
    validate_structure_selection,
)


EVENT_PREFIX = "CRYOID_EVENT "

CONFIGURABLE_KEYS = {
    "project-dir": "project_dir",
    "weights-dir": "weights_dir",
    "device": "device",
    "rcl": "rcl",
    "protein-min-trace-length": "protein_min_trace_length",
    "structure-selection": "structure_selection",
    "keep-tmp": "keep_tmp",
    "chimerax-path": "chimerax_path",
    "mode": "sequence_mode",
    "blast-bin-dir": "blast_bin_dir",
}


def format_command(command: Sequence[str]) -> str:
    """Format an argument vector for terminal and pipeline-log display."""

    if os.name == "nt":
        import subprocess

        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def encode_event(event_type: str, **payload: Any) -> str:
    return EVENT_PREFIX + json.dumps({"type": event_type, **payload}, separators=(",", ":"))


def decode_event(line: str) -> dict[str, Any] | None:
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        payload = json.loads(line[len(EVENT_PREFIX) :])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _result_exit_code(result: PipelineResult) -> int:
    statuses = {str(stage.status) for stage in result.stages}
    if "error" in statuses:
        return 1
    if "skipped" in statuses:
        return 130
    if any(status.startswith("blocked") for status in statuses):
        return 2
    return 0


def _selected_config_path(args: argparse.Namespace) -> Path:
    return Path(args.config).expanduser().resolve() if args.config is not None else default_config_path()


def _config_path(args: argparse.Namespace, _raw_argv: Sequence[str]) -> int:
    print(_selected_config_path(args))
    return 0


def _config_show(args: argparse.Namespace, _raw_argv: Sequence[str]) -> int:
    path = _selected_config_path(args)
    config = load_config(path)
    print(f"Config file: {path}")
    print(f"Exists: {'yes' if path.is_file() else 'no'}")
    for public_key, stored_key in CONFIGURABLE_KEYS.items():
        print(f"{public_key} = {config.get(stored_key, '')}")
    return 0


def _config_set(args: argparse.Namespace, _raw_argv: Sequence[str]) -> int:
    stored_key = CONFIGURABLE_KEYS[args.key]
    path = _selected_config_path(args)
    value = args.value
    if args.key == "blast-bin-dir":
        try:
            value = str(resolve_blast_bin_dir(args.value, {}))
        except ValueError as exc:
            print(f"Could not set blast-bin-dir: {exc}", file=sys.stderr)
            return 2
    elif args.key == "protein-min-trace-length":
        try:
            value = str(resolve_protein_min_trace_length(args.value, {}))
        except ValueError as exc:
            print(f"Could not set protein-min-trace-length: {exc}", file=sys.stderr)
            return 2
    elif args.key == "structure-selection":
        try:
            value = validate_structure_selection(args.value)
        except ValueError as exc:
            print(f"Could not set structure-selection: {exc}", file=sys.stderr)
            return 2
    elif args.key == "keep-tmp":
        try:
            value = "true" if validate_keep_tmp(args.value) else "false"
        except ValueError as exc:
            print(f"Could not set keep-tmp: {exc}", file=sys.stderr)
            return 2
    update_config(path, **{stored_key: value})
    print(f"Set {args.key} = {value}")
    print(f"Config file: {path}")
    return 0


def _config_unset(args: argparse.Namespace, _raw_argv: Sequence[str]) -> int:
    stored_key = CONFIGURABLE_KEYS[args.key]
    path = _selected_config_path(args)
    update_config(path, **{stored_key: ""})
    print(f"Unset {args.key}")
    print(f"Config file: {path}")
    return 0


def _run_pipeline(args: argparse.Namespace, raw_argv: Sequence[str]) -> int:
    user_config = load_config(args.config)
    project_dir = resolve_project_dir(args.project_dir, user_config)
    weights_dir = resolve_weights_dir(args.weights_dir, user_config)
    device = resolve_device(args.device, user_config)
    rcl = resolve_rcl(args.rcl, user_config)
    protein_min_trace_length = resolve_protein_min_trace_length(
        args.protein_min_trace_length, user_config
    )
    sequence_mode = resolve_sequence_mode(args.mode, user_config)
    structure_selection = resolve_structure_selection(args.structure_selection, user_config)
    keep_tmp = resolve_keep_tmp(args.keep_tmp, user_config)
    blast_bin_dir = resolve_blast_bin_dir(args.blast_bin_dir, user_config)
    cancellation_requested = threading.Event()
    cancel_file = Path(args.cancel_file).expanduser().resolve() if args.cancel_file else None

    def request_cancel(_signum: int, _frame: object) -> None:
        cancellation_requested.set()
        print("Cancellation requested; waiting for the current pipeline checkpoint.", flush=True)

    previous_handlers: dict[int, Any] = {}
    for signal_name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signal_name, None)
        if signum is not None:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_cancel)

    command = [sys.executable, "-m", "cryoid_pipeline", *raw_argv]
    command_line = format_command(command)

    def progress(message: str) -> None:
        print(str(message), flush=True)

    def progress_percent(value: int) -> None:
        if args.event_stream:
            print(encode_event("progress", value=int(value)), flush=True)
        else:
            print(f"Progress: {int(value)}%", flush=True)

    def should_cancel() -> bool:
        return cancellation_requested.is_set() or bool(cancel_file and cancel_file.exists())

    try:
        pipeline = CryoIDPipeline(
            repo_root=args.repo_root,
            project_dir=project_dir,
            unet_weights_dir=weights_dir,
            unet_device=device,
            unet_rcl=rcl,
            protein_min_trace_length=protein_min_trace_length,
            sequence_mode=sequence_mode,
            structure_selection=structure_selection,
            keep_tmp=keep_tmp,
            blast_bin_dir=blast_bin_dir,
        )
        result = pipeline.run(
            args.density_map,
            args.candidate_pool,
            progress=progress,
            progress_percent=progress_percent,
            should_cancel=should_cancel,
            command_line=command_line,
            stop_after_identification=args.stop_after_identification,
        )
        if args.event_stream:
            print(encode_event("result", result=pipeline_result_to_dict(result)), flush=True)
        exit_code = _result_exit_code(result)
        status_label = {0: "complete", 1: "error", 2: "blocked", 130: "cancelled"}.get(exit_code, "unknown")
        print(f"Run directory: {result.run_dir}", flush=True)
        print(f"Run status: {status_label}", flush=True)
        return exit_code
    finally:
        if cancel_file is not None:
            cancel_file.unlink(missing_ok=True)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cryoid", description="Run the CryoID pipeline without the GUI.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run the same end-to-end pipeline used by the GUI.")
    run_parser.add_argument("density_map", type=Path, help="Input .map/.mrc/.ccp4 density map.")
    run_parser.add_argument(
        "candidate_pool",
        type=Path,
        help="Candidate protein or RNA sequence pool in FASTA format.",
    )
    run_parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help=(
            "Output parent directory; CryoID creates or reuses its cryoid_results child. "
            "Defaults to CRYOID_PROJECT_DIR, config.toml, then the current directory."
        ),
    )
    run_parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="UNet weights directory; defaults to CRYOID_UNET_WEIGHTS_DIR, then config.toml.",
    )
    run_parser.add_argument(
        "--device",
        default=None,
        help="Torch device, such as cuda:0 or cpu; defaults to environment, config, then auto-detection.",
    )
    run_parser.add_argument(
        "--rcl",
        type=float,
        default=None,
        help=(
            "Recommended contour level used for UNet map normalization; zero is "
            "used literally. The archived EMD-47655 RNA test used 0.05."
        ),
    )
    run_parser.add_argument(
        "--mode",
        choices=("auto", "protein", "rna"),
        default=None,
        help="Sequence pipeline mode; defaults to config, then auto-detection.",
    )
    run_parser.add_argument(
        "--protein-min-trace-length",
        type=int,
        default=None,
        help="Minimum number of traced C-alpha nodes retained as a protein query; defaults to 10.",
    )
    run_parser.add_argument(
        "--structure-selection",
        default=None,
        help=(
            "Top ranked sequences sent to structure retrieval/fitting. A positive integer "
            "selects a count (1 means one); a decimal from 0 to 1 selects a fraction "
            "(1.0 means 100%%). Defaults to 1 for protein and 0.5 for RNA."
        ),
    )
    run_parser.add_argument(
        "--blast-bin-dir",
        type=Path,
        default=None,
        help=(
            "Folder containing blastp, blastn, and makeblastdb; defaults to config, "
            "CRYOID_BLAST_BIN_DIR, the active environment, or PATH."
        ),
    )
    run_parser.add_argument(
        "--keep-tmp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retain all run temporary files and tested models; defaults to false.",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Alternate TOML config file; defaults to CRYOID_CONFIG or the user config directory.",
    )
    run_parser.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    run_parser.add_argument(
        "--stop-after-identification",
        action="store_true",
        help=(
            "Stop after map-derived query generation and candidate ranking; "
            "do not fetch or fit structures."
        ),
    )
    run_parser.add_argument("--event-stream", action="store_true", help=argparse.SUPPRESS)
    run_parser.add_argument("--cancel-file", type=Path, default=None, help=argparse.SUPPRESS)
    run_parser.set_defaults(handler=_run_pipeline)

    config_parser = subparsers.add_parser("config", help="View or change persistent GUI/CLI defaults.")
    config_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Alternate TOML config file; defaults to CRYOID_CONFIG or the user config directory.",
    )
    config_actions = config_parser.add_subparsers(dest="config_action", required=True)

    path_parser = config_actions.add_parser("path", help="Print the config file location.")
    path_parser.set_defaults(handler=_config_path)

    show_parser = config_actions.add_parser("show", help="Show persistent defaults.")
    show_parser.set_defaults(handler=_config_show)

    set_parser = config_actions.add_parser("set", help="Save a persistent default.")
    set_parser.add_argument("key", choices=CONFIGURABLE_KEYS, help="Setting to save.")
    set_parser.add_argument("value", help="Value to save.")
    set_parser.set_defaults(handler=_config_set)

    unset_parser = config_actions.add_parser("unset", help="Clear a persistent default.")
    unset_parser.add_argument("key", choices=CONFIGURABLE_KEYS, help="Setting to clear.")
    unset_parser.set_defaults(handler=_config_unset)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    return int(args.handler(args, raw_argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVENT_PREFIX",
    "build_parser",
    "decode_event",
    "encode_event",
    "format_command",
    "main",
    "pipeline_result_from_dict",
]
