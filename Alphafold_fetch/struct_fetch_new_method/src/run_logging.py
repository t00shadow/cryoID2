"""
Stdout/stderr tee logging used by all CLI entry points.

:func:`tee_output` returns a context manager that redirects stdout and stderr
to both the original streams and an append-mode log file under the run
directory. The same log file ends with timestamped start/end markers and the
traceback of any exception that escapes the context.
"""
from __future__ import annotations

import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO


class TeeStream:
    """File-like wrapper that mirrors writes to a base stream and a log handle.

    Failures on the base stream (e.g. closed stdout under daemonization)
    silently disable further base-stream writes while keeping the log
    file in sync, so the run never crashes on stdout I/O errors.
    """

    def __init__(self, stream: TextIO, log_handle: TextIO):
        self._stream = stream
        self._log_handle = log_handle
        self._stream_ok = True
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, text: str) -> int:
        """Write ``text`` to the base stream (best-effort) and the log handle."""
        if self._stream_ok:
            try:
                self._stream.write(text)
            except (OSError, ValueError):
                self._stream_ok = False
        self._log_handle.write(text)
        self.flush()
        return len(text)

    def flush(self) -> None:
        """Flush both the base stream (best-effort) and the log handle."""
        if self._stream_ok:
            try:
                self._stream.flush()
            except (OSError, ValueError):
                self._stream_ok = False
        self._log_handle.flush()

    def isatty(self) -> bool:
        """Proxy the base stream's tty status."""
        return bool(getattr(self._stream, "isatty", lambda: False)())


@contextmanager
def tee_output(log_path: Path) -> Iterator[Path]:
    """Mirror stdout/stderr to ``log_path`` (append mode) for the duration of the block.

    Any exception that escapes the block is logged with full traceback before
    being re-raised, and the log file always ends with a timestamped end
    marker even on crash.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        started = datetime.now().isoformat(timespec="seconds")
        handle.write(f"\n--- terminal log started {started} ---\n")
        handle.flush()
        try:
            try:
                with redirect_stdout(TeeStream(sys.stdout, handle)), redirect_stderr(TeeStream(sys.stderr, handle)):
                    yield log_path
            except BaseException:
                traceback.print_exc(file=handle)
                handle.flush()
                raise
        finally:
            ended = datetime.now().isoformat(timespec="seconds")
            handle.write(f"\n--- terminal log ended {ended} ---\n")
            handle.flush()
