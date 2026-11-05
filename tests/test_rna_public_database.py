from __future__ import annotations

import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from I_Structure_fitting.rna.structures import ensure_public_rna_database


class FakeResponse:
    def __init__(self, content: bytes = b"", headers: dict[str, str] | None = None) -> None:
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, _size: int):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class PublicRnaDatabaseTests(unittest.TestCase):
    @staticmethod
    def _cached_project(root: Path) -> Path:
        project = root / "project"
        cache = project / "cache" / "rna_structures"
        snapshot = cache / "snapshots" / "old"
        snapshot.mkdir(parents=True)
        (snapshot / "pdb_seqres.txt.gz").write_bytes(b"old archive")
        (snapshot / "pdb_rna_chains.fasta").write_text(">pdb|1ABC|A\nACGT\n", encoding="ascii")
        (snapshot / "pdb_rna_chains.nin").write_bytes(b"db")
        (cache / "database_metadata.json").write_text(
            json.dumps(
                {
                    "active_snapshot": "old",
                    "remote": {"etag": "old", "last_modified": "yesterday"},
                }
            ),
            encoding="utf-8",
        )
        return project

    @staticmethod
    def _subprocess(command, **kwargs):
        if "makeblastdb" in str(command[0]):
            for flag in ("-in", "-out"):
                value = str(command[command.index(flag) + 1])
                assert not any(char.isspace() for char in value), (flag, value)

            prefix = Path(command[command.index("-out") + 1])

            if not prefix.is_absolute():
                prefix = Path(kwargs.get("cwd") or Path.cwd()) / prefix

            prefix.with_suffix(".nin").write_bytes(b"database")
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        return SimpleNamespace(
            stdout="blastn: 2.16.0+\n",
            stderr="",
            returncode=0,
        )

    def test_first_download_builds_a_versioned_validated_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = io.BytesIO()
            with gzip.GzipFile(fileobj=payload, mode="wb") as handle:
                handle.write(b">pdb|1ABC|A mol:na RNA (4-MER)\nACGU\n")
            head = FakeResponse(headers={"ETag": "new", "Last-Modified": "today"})
            get = FakeResponse(payload.getvalue())
            with (
                patch("I_Structure_fitting.rna.structures.requests.head", return_value=head),
                patch("I_Structure_fitting.rna.structures.requests.get", return_value=get),
                patch("I_Structure_fitting.rna.structures._executable", side_effect=lambda name, *_args: name),
                patch("I_Structure_fitting.rna.structures.subprocess.run", side_effect=self._subprocess),
            ):
                result = ensure_public_rna_database(
                    root / "RNA for Star", progress=lambda _message: None, check_cancel=lambda: None
                )
            self.assertTrue(result.fasta_path.is_file())
            self.assertIn("ACGT", result.fasta_path.read_text(encoding="ascii"))
            self.assertEqual(result.metadata["remote"]["etag"], "new")
            self.assertTrue(result.metadata["active_snapshot"])

    def test_offline_run_reuses_last_validated_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = self._cached_project(Path(tmp))
            with (
                patch("I_Structure_fitting.rna.structures.requests.head", side_effect=requests.ConnectionError("offline")),
                patch("I_Structure_fitting.rna.structures._executable", return_value="blastn"),
                patch("I_Structure_fitting.rna.structures.subprocess.run", return_value=SimpleNamespace(stdout="blastn 2\n")),
            ):
                result = ensure_public_rna_database(
                    project, progress=lambda _message: None, check_cancel=lambda: None
                )
            self.assertEqual(result.metadata["active_snapshot"], "old")
            self.assertTrue(result.warnings)

    def test_corrupt_refresh_does_not_replace_valid_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = self._cached_project(Path(tmp))
            head = FakeResponse(headers={"ETag": "changed", "Last-Modified": "today"})
            with (
                patch("I_Structure_fitting.rna.structures.requests.head", return_value=head),
                patch("I_Structure_fitting.rna.structures.requests.get", return_value=FakeResponse(b"not gzip")),
                patch("I_Structure_fitting.rna.structures._executable", return_value="blastn"),
                patch("I_Structure_fitting.rna.structures.subprocess.run", return_value=SimpleNamespace(stdout="blastn 2\n")),
            ):
                result = ensure_public_rna_database(
                    project, progress=lambda _message: None, check_cancel=lambda: None
                )
            self.assertEqual(result.metadata["active_snapshot"], "old")
            self.assertEqual(result.metadata["remote"]["etag"], "old")
            self.assertTrue(result.warnings)

    def test_offline_without_cache_fails_only_the_structure_stage_caller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("I_Structure_fitting.rna.structures.requests.head", side_effect=requests.ConnectionError("offline")),
                patch("I_Structure_fitting.rna.structures.requests.get", side_effect=requests.ConnectionError("offline")),
            ):
                with self.assertRaises(requests.ConnectionError):
                    ensure_public_rna_database(
                        Path(tmp) / "project",
                        progress=lambda _message: None,
                        check_cancel=lambda: None,
                    )


if __name__ == "__main__":
    unittest.main()
