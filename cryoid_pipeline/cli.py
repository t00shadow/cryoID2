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
from .user_config import (
    PROJECT_STATE_KEYS, STARTUP_KEYS,
    DEFAULT_CONFIG,
    default_config_path,
    load_config,
    load_effective_config,
    project_config_path,
    results_project_dir,
    resolve_run_settings,
    resolve_chimerax_path,
    resolve_weights_dir,
    save_config, save_config_changes, PATH_KEYS, normalize_path,
    update_config,
    validate_config_value,
)


EVENT_PREFIX = "CRYOID_EVENT "

CONFIGURABLE_KEYS = {
    "weights-dir": "weights_dir",
    "device": "device",
    "cpu-threads": "cpu_threads",
    "rcl": "rcl",
    "protein-min-trace-length": "protein_min_trace_length",
    "structure-selection": "structure_selection",
    "keep-tmp": "keep_tmp",
    "chimerax-path": "chimerax_path",
    "mode": "sequence_mode",
    "map": "last_density_map",
    "pool": "last_candidate_pool",
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
    if args.project_dir is not None:
        return project_config_path(results_project_dir(args.project_dir))
    return Path(args.config).expanduser().resolve() if args.config is not None else default_config_path()


def _config_values(args: argparse.Namespace) -> dict[str, str]:
    if args.project_dir is not None:
        loaded = load_effective_config(args.project_dir)
        values = dict(loaded.values)
        if loaded.source_path is None:
            for key in PROJECT_STATE_KEYS:
                values[key] = ""
        return values
    return load_config(_selected_config_path(args))


def _saved_value(key: str, value: Any) -> str:
    if key in PATH_KEYS and value:
        return normalize_path(value)
    return validate_config_value(key, value)


def _save_setting(args: argparse.Namespace, key: str, value: str) -> None:
    path = _selected_config_path(args)
    if args.project_dir is None:
        update_config(path, **{key: value})
    else:
        values = _config_values(args)
        save_config_changes({key: value}, path, initial=values, project=True)


def _init_project(args: argparse.Namespace, _raw_argv: Sequence[str]) -> int:
    project = results_project_dir(args.project_path)
    path = project_config_path(project)
    if path.exists():
        raise ValueError(f"Project is already initialized: {project}\nUse 'cryoid config --project PATH' to view or edit it.")
    loaded = load_effective_config(project, config_path=args.config)
    values = dict(loaded.values)
    if loaded.source_path is None:
        for key in PROJECT_STATE_KEYS:
            values[key] = ""
    for public, key in CONFIGURABLE_KEYS.items():
        if public in {"map", "pool"}:
            continue
        if key in STARTUP_KEYS:
            continue
        explicit = getattr(args, public.replace("-", "_"), None)
        if explicit is not None:
            values[key] = explicit
        if key == "structure_selection" and not str(values[key]).strip():
            continue
        values[key] = _saved_value(key, values[key])
    for argument, key in ((args.map_path, "last_density_map"), (args.pool_path, "last_candidate_pool")):
        if argument is not None:
            values[key] = _saved_value(key, argument)
    save_config(values, path, overwrite=False, project=True)
    print(f"Project initialized: {project}")
    print(f"Config file: {path}")
    return 0


def _config_path(args: argparse.Namespace, _raw_argv: Sequence[str]) -> int:
    print(_selected_config_path(args))
    return 0


def _config_show(args: argparse.Namespace, _raw_argv: Sequence[str]) -> int:
    path = _selected_config_path(args)
    config = _config_values(args)
    print(f"Config file: {path}")
    print(f"Exists: {'yes' if path.is_file() else 'no'}")
    for public_key, stored_key in CONFIGURABLE_KEYS.items():
        if args.project_dir is not None and stored_key in STARTUP_KEYS:
            continue
        print(f"{public_key} = {config.get(stored_key, '')}")
    return 0


def _config_set(args: argparse.Namespace, _raw_argv: Sequence[str]) -> int:
    stored_key = CONFIGURABLE_KEYS[args.key]
    path = _selected_config_path(args)
    try:
        if args.project_dir is not None and stored_key in STARTUP_KEYS:
            raise ValueError(f"{args.key} is a global startup setting. Use cryoid config set {args.key} VALUE without --project.")
        value = _saved_value(stored_key, args.value)
    except ValueError as exc:
        print(f"Could not set {args.key}: {exc}", file=sys.stderr)
        return 2
    _save_setting(args, stored_key, value)
    print(f"Set {args.key} = {value}")
    print(f"Config file: {path}")
    return 0


def _config_unset(args: argparse.Namespace, _raw_argv: Sequence[str]) -> int:
    stored_key = CONFIGURABLE_KEYS[args.key]
    path = _selected_config_path(args)
    if args.project_dir is not None and stored_key in STARTUP_KEYS:
        raise ValueError(f"{args.key} is a global startup setting. Use cryoid config unset {args.key} without --project.")
    _save_setting(args, stored_key, "")
    print(f"Unset {args.key}")
    print(f"Config file: {path}")
    return 0


def _run_pipeline(args: argparse.Namespace, raw_argv: Sequence[str]) -> int:
    selected_project = args.project_dir if args.project_dir is not None else Path.cwd()
    loaded = load_effective_config(
        selected_project, config_path=args.config,
        user_config=DEFAULT_CONFIG if args.no_config else None,
        use_project_config=not args.no_config,
    )
    project_dir, user_config = loaded.project_dir, loaded.values

    def input_path(named: Path | None, key: str, option: str) -> Path:
        if named is not None:
            return named
        saved = user_config.get(key, "") if loaded.source_path is not None else ""
        if not saved:
            raise ValueError(f"No input saved for {option}. Supply '{option} PATH'.")
        path = Path(saved).expanduser()
        return path if path.is_absolute() else (loaded.source_path.parent / path).resolve()

    density_map = input_path(args.map_path, "last_density_map", "--map")
    candidate_pool = input_path(args.pool_path, "last_candidate_pool", "--pool")
    weights_argument = args.weights_dir if args.weights_dir is not None else ("" if args.no_config else None)
    weights_dir = resolve_weights_dir(weights_argument, user_config)
    chimerax_argument = args.chimerax_path if args.chimerax_path is not None else ("" if args.no_config else None)
    chimerax_path = resolve_chimerax_path(user_config, argument=chimerax_argument)
    overrides = {
        "device": args.device, "cpu_threads": args.cpu_threads, "rcl": args.rcl,
        "protein_min_trace_length": args.protein_min_trace_length,
        "sequence_mode": args.mode, "structure_selection": args.structure_selection,
        "keep_tmp": args.keep_tmp,
    }
    if args.no_config:
        overrides = {key: (value if value is not None else DEFAULT_CONFIG[key]) for key, value in overrides.items()}
        if not overrides["structure_selection"]:
            overrides["structure_selection"] = "auto"
    settings, _ = resolve_run_settings(user_config, strict=True, overrides=overrides)
    path = project_config_path(project_dir)
    if not path.exists():
        initial = dict(user_config)
        if loaded.source_path is None:
            for key in PROJECT_STATE_KEYS:
                initial[key] = ""
        initial.update(
            settings,
            project_dir=str(project_dir),
            weights_dir=str(weights_dir or ""),
            chimerax_path=chimerax_path,
            device=settings["device"] or "auto",
            cpu_threads=settings["cpu_threads"] if settings["cpu_threads"] is not None else "auto",
            structure_selection=settings["structure_selection"] or "",
            keep_tmp="true" if settings["keep_tmp"] else "false",
            last_density_map=str(density_map.expanduser().resolve()),
            last_candidate_pool=str(candidate_pool.expanduser().resolve()),
        )
        try:
            save_config(initial, path, overwrite=False, project=True)
        except FileExistsError:
            pass
        print(f"Created project config: {path}", flush=True)
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

    def status(event: dict[str, Any]) -> None:
        if (event.get("weights_path") and event.get("automatic")
                and not args.no_config and args.weights_dir is None):
            # The GUI uses --no-config and persists through its own conflict-
            # aware settings handler. A CLI run saves to its selected user file.
            try:
                baseline = load_config(args.config)
                if baseline["weights_dir"] == user_config.get("weights_dir", ""):
                    save_config_changes({"weights_dir": event["weights_path"]},
                                        path=args.config, baseline=baseline)
                    progress(f"Weights: saved location to {args.config or default_config_path()}")
                else:
                    progress("Weights: startup configuration changed during this run; keeping the newer weights setting.")
            except (OSError, ValueError, TimeoutError) as exc:
                progress(f"Weights: could not save the downloaded weights location: {exc}")
        if args.event_stream:
            print(encode_event("status", **event), flush=True)

    try:
        from .orchestrator import CryoIDPipeline

        pipeline = CryoIDPipeline(
            project_dir=project_dir,
            unet_weights_dir=weights_dir if weights_dir is not None else "",
            unet_device=settings["device"] or "auto",
            cpu_threads=settings["cpu_threads"] if settings["cpu_threads"] is not None else "auto",
            unet_rcl=settings["rcl"],
            protein_min_trace_length=settings["protein_min_trace_length"],
            sequence_mode=settings["sequence_mode"],
            structure_selection=settings["structure_selection"],
            keep_tmp=settings["keep_tmp"],
            chimerax_path=chimerax_path,
            user_config=user_config,
        )
        result = pipeline.run(
            density_map,
            candidate_pool,
            progress=progress,
            progress_percent=progress_percent,
            status=status,
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


def _add_run_options(parser: argparse.ArgumentParser, *, startup_paths: bool = True) -> None:
    if startup_paths:
        parser.add_argument(
            "--weights-dir",
            type=str,
            default=None,
            help="Custom weights directory for this run. Empty or missing paths autodownload published weights.",
        )
        parser.add_argument("--chimerax-path", default=None,
                            help="Override the global startup ChimeraX path for this run; an empty string disables it.")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device: auto, cpu, mps, or cuda:N. Auto prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument(
        "--cpu-threads", default=None, metavar="auto|N",
        help="CPU threads per supported numerical pool and BLAST search. Auto preserves library defaults.",
    )
    parser.add_argument(
        "--rcl",
        type=float,
        default=None,
        help=(
            "Recommended contour level used for UNet map normalization; "
            "defaults to zero, which is used literally."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "protein", "rna"),
        default=None,
        help="Sequence pipeline mode; defaults to config, then auto-detection.",
    )
    parser.add_argument(
        "--protein-min-trace-length",
        type=int,
        default=None,
        help="Minimum number of traced C-alpha nodes retained as a protein query; defaults to 10.",
    )
    parser.add_argument(
        "--structure-selection",
        default=None,
        help=(
            "Top ranked sequences sent to structure retrieval/fitting. A positive integer "
            "selects a count (1 means one); a decimal from 0 to 1 selects a fraction "
            "(1.0 means 100%%). Use auto for mode defaults: 1 for protein and 0.5 for RNA."
        ),
    )
    parser.add_argument(
        "--keep-tmp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retain all run temporary files and tested models; defaults to false.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="User defaults TOML file; project settings and explicit options take precedence. Defaults to CRYOID_CONFIG or the user config directory.",
    )


def _add_config_target(parser: argparse.ArgumentParser, *, default: Any = None) -> None:
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--project", dest="project_dir", type=Path, default=default,
                        help="View or edit this project's saved settings.")
    target.add_argument("--config", type=Path, default=default,
                        help="Config file path; otherwise use global user defaults.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryoid", description="Create projects and run CryoID with saved settings.",
        epilog="First run here: cryoid run --map MAP --pool FASTA\n"
               "Run again: cryoid run --project PROJECT\n"
               "Settings: cryoid config --project PROJECT\n"
               "Optional setup without running: cryoid init PROJECT --map MAP --pool FASTA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init", help="Create a project and save its inputs and settings.")
    init_parser.add_argument("project_path", type=Path, metavar="PROJECT", help="Project parent or cryoid_results directory.")
    init_parser.add_argument("--map", dest="map_path", type=Path, help="Save the input density map.")
    init_parser.add_argument("--pool", dest="pool_path", type=Path, help="Save the candidate FASTA.")
    _add_run_options(init_parser, startup_paths=False)
    init_parser.set_defaults(handler=_init_project)
    run_parser = subparsers.add_parser("run", help="Run the same end-to-end pipeline used by the GUI.")
    run_parser.add_argument(
        "--project", dest="project_dir",
        type=Path,
        default=None,
        help=(
            "Output parent directory; CryoID creates or reuses its cryoid_results child. "
            "Loads or creates that project's config.toml. Defaults to the current directory."
        ),
    )
    run_parser.add_argument("--map", dest="map_path", type=Path, help="Override the saved density map for this run.")
    run_parser.add_argument("--pool", dest="pool_path", type=Path, help="Override the saved candidate FASTA for this run.")
    _add_run_options(run_parser)
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
    run_parser.add_argument("--no-config", action="store_true",
                            help="Use only supplied arguments and built-in defaults; ignore saved settings and environment overrides.")
    run_parser.set_defaults(handler=_run_pipeline)

    config_parser = subparsers.add_parser("config", help="View or edit project settings or global defaults.",
                                         description="View settings, or use set/unset to edit them. Without an action, show settings.")
    _add_config_target(config_parser)
    config_parser.set_defaults(handler=_config_show)
    config_actions = config_parser.add_subparsers(dest="config_action")

    path_parser = config_actions.add_parser("path", help="Print the config file location.")
    path_parser.set_defaults(handler=_config_path)

    show_parser = config_actions.add_parser("show", help="Show saved settings.")
    show_parser.set_defaults(handler=_config_show)

    set_parser = config_actions.add_parser("set", help="Save a setting.")
    set_parser.add_argument("key", choices=CONFIGURABLE_KEYS, help="Setting to save.")
    set_parser.add_argument("value", help="Value to save.")
    set_parser.set_defaults(handler=_config_set)

    unset_parser = config_actions.add_parser("unset", help="Clear a setting.")
    unset_parser.add_argument("key", choices=CONFIGURABLE_KEYS, help="Setting to clear.")
    unset_parser.set_defaults(handler=_config_unset)
    for action in (path_parser, show_parser, set_parser, unset_parser):
        _add_config_target(action, default=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if args.command == "config" and args.config is not None and args.project_dir is not None:
        parser.error("Choose --project or --config, not both.")
    if args.command == "run":
        if args.no_config and args.config is not None:
            parser.error("Choose --no-config or --config, not both.")
    try:
        return int(args.handler(args, raw_argv))
    except (ValueError, OSError) as exc:
        print(f"Cannot continue: {exc}", file=sys.stderr)
        return 2


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
