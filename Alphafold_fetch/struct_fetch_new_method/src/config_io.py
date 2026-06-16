"""
JSON-config loading and CLI-override merging for pipeline and benchmark scripts.

Every entry point follows the same pattern:

1. :func:`resolve_method_path` turns a string into a concrete ``Path``, trying
   ``METHOD_ROOT``-relative and CWD-relative interpretations.
2. :func:`load_json_config` reads the JSON config and validates that every key
   is either prefixed with ``_`` (treated as a comment) or appears in the
   caller-supplied ``defaults`` dict.
3. :func:`apply_cli_overrides` overrides any default whose corresponding CLI
   flag was provided (i.e. value is not ``None``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paths import METHOD_ROOT


# Top-level directories that are always interpreted as METHOD_ROOT-relative
# when no absolute or existing CWD-relative path matches.
METHOD_RELATIVE_ROOTS = {"configs", "runs", "test_data", "scripts", "src"}


def resolve_method_path(value: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a CLI/config path against ``METHOD_ROOT`` and the current working dir.

    Absolute paths are returned as-is (resolved). For relative paths the
    candidate that exists wins; if neither exists, paths whose first part
    names a known package directory (see :data:`METHOD_RELATIVE_ROOTS`) are
    interpreted as ``METHOD_ROOT``-relative, otherwise as CWD-relative.
    Raises ``FileNotFoundError`` when ``must_exist`` is True and nothing matches.
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError("Cannot resolve an empty path.")
    raw_path = Path(text).expanduser()
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        method_path = (METHOD_ROOT / raw_path).resolve()
        cwd_path = (Path.cwd() / raw_path).resolve()
        if must_exist:
            if method_path.exists():
                resolved = method_path
            elif cwd_path.exists():
                resolved = cwd_path
            else:
                raise FileNotFoundError(f"Path does not exist: {text} (tried {method_path} and {cwd_path})")
        elif raw_path.parts and raw_path.parts[0].lower() in METHOD_RELATIVE_ROOTS:
            resolved = method_path
        else:
            resolved = cwd_path
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def load_json_config(path: str | Path, defaults: dict[str, Any]) -> dict[str, Any]:
    """Read a JSON config, validate its keys against ``defaults``, and return merged settings.

    Keys prefixed with ``_`` (e.g. ``"_description"``) are treated as inline
    comments and ignored. Any other unknown key raises ``ValueError`` so typos
    surface early instead of silently being ignored.
    """
    config_path = resolve_method_path(path, must_exist=True)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a JSON object: {config_path}")
    unknown = sorted(key for key in payload if not key.startswith("_") and key not in defaults)
    if unknown:
        raise ValueError(f"Unknown config key(s) in {config_path}: {', '.join(unknown)}")
    settings = dict(defaults)
    settings.update({key: value for key, value in payload.items() if not key.startswith("_")})
    settings["config"] = str(config_path)
    return settings


def apply_cli_overrides(settings: dict[str, Any], args: Any, *, skip_keys: set[str] | None = None) -> dict[str, Any]:
    """Override ``settings`` with any non-None values from an ``argparse.Namespace``.

    The ``config`` key is always preserved from ``settings``. Callers can pass
    extra keys via ``skip_keys`` to protect them from CLI overrides.
    """
    out = dict(settings)
    ignored = {"config"} | set(skip_keys or set())
    for key, value in vars(args).items():
        if key in ignored or value is None:
            continue
        out[key] = value
    return out
