from __future__ import annotations

from pathlib import Path

import pytest

from G_GUI import launcher


def _create_qt_libraries(prefix: Path) -> Path:
    library_dir = prefix / "lib"
    library_dir.mkdir()
    for name in launcher._LINUX_QT_LIBRARIES:
        (library_dir / name).touch()
    return library_dir


def test_linux_launcher_reexecutes_with_conda_library_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library_dir = _create_qt_libraries(tmp_path)
    captured: dict[str, object] = {}

    def fake_execve(
        executable: str,
        arguments: list[str],
        environment: dict[str, str],
    ) -> None:
        captured.update(
            executable=executable,
            arguments=arguments,
            environment=environment,
        )
        raise RuntimeError("re-executed")

    monkeypatch.setattr(launcher.sys, "platform", "linux")
    monkeypatch.setattr(launcher.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(launcher.sys, "argv", ["cryoid-gui", "--example"])
    monkeypatch.setattr(launcher.os, "execve", fake_execve)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    monkeypatch.delenv(launcher._LINUX_LIBRARY_PATH_READY, raising=False)

    with pytest.raises(RuntimeError, match="re-executed"):
        launcher._ensure_linux_qt_library_path()

    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment["LD_LIBRARY_PATH"] == f"{library_dir}:/existing/lib"
    assert environment[launcher._LINUX_LIBRARY_PATH_READY] == "1"
    assert captured["arguments"] == [
        launcher.sys.executable,
        "-m",
        "G_GUI.launcher",
        "--example",
    ]


def test_linux_launcher_does_not_reexecute_when_path_is_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _create_qt_libraries(tmp_path)
    library_dir = Path("/conda/environment/lib")

    monkeypatch.setattr(launcher.sys, "platform", "linux")
    monkeypatch.setattr(launcher, "_conda_qt_library_dir", lambda: library_dir)
    monkeypatch.setattr(
        launcher.os,
        "execve",
        lambda *_args: pytest.fail("launcher re-executed unexpectedly"),
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", str(library_dir))

    launcher._ensure_linux_qt_library_path()
