from __future__ import annotations

import logging
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from D_BLAST_searching.blast_tools import (
    BLAST_PROGRAMS,
    blast_executable,
    bundled_blast_bin_dir,
    validate_blast_bin_dir,
)
from D_BLAST_searching.reduced_alphabet.search_support import QueryMap


class BlastToolTests(unittest.TestCase):
    @staticmethod
    def _fake_bin(root: Path, *, system: str) -> Path:
        directory = root / ("Scripts" if system == "windows" else "bin")
        directory.mkdir(parents=True)
        suffix = ".exe" if system == "windows" else ""
        for program in BLAST_PROGRAMS:
            executable = directory / f"{program}{suffix}"
            executable.write_text("fake", encoding="ascii")
            if system != "windows":
                executable.chmod(0o755)
        return directory

    def test_validates_windows_and_linux_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            windows = self._fake_bin(root / "windows", system="windows")
            linux = self._fake_bin(root / "linux", system="linux")
            self.assertEqual(validate_blast_bin_dir(windows, system="windows"), windows.resolve())
            self.assertEqual(validate_blast_bin_dir(linux, system="linux"), linux.resolve())

    def test_bundle_is_selected_despite_obsolete_paths_and_environment(self) -> None:
        expected = bundled_blast_bin_dir()
        with patch.dict(os.environ, {"PATH": "", "CRYOID_BLAST_BIN_DIR": "missing"}):
            self.assertEqual(bundled_blast_bin_dir(), expected)
            for program in BLAST_PROGRAMS:
                self.assertEqual(Path(blast_executable(program)).parent, expected)

    def test_supported_platforms_and_architecture_aliases(self) -> None:
        from D_BLAST_searching.blast_tools import BUNDLE_ROOT, bundled_blast_bin_dir
        for system in ("windows", "linux", "darwin"):
            for machine in ("AMD64", "x86_64", "x64"):
                self.assertEqual(bundled_blast_bin_dir(system=system, machine=machine),
                                 (BUNDLE_ROOT / f"{system}-x86_64" / "bin").resolve())
        for machine in ("arm64", "aarch64"):
            self.assertEqual(bundled_blast_bin_dir(system="darwin", machine=machine),
                             (BUNDLE_ROOT / "darwin-arm64" / "bin").resolve())
        for system, machine in (("windows", "arm64"), ("linux", "aarch64"), ("darwin", "i386")):
            with self.assertRaisesRegex(RuntimeError, "support"):
                bundled_blast_bin_dir(system=system, machine=machine)

    def test_missing_bundle_never_falls_back_to_system_blast(self) -> None:
        from D_BLAST_searching.blast_tools import bundled_blast_bin_dir
        with tempfile.TemporaryDirectory() as tmp:
            with patch("D_BLAST_searching.blast_tools.BUNDLE_ROOT", Path(tmp)):
                with self.assertRaisesRegex(FileNotFoundError, "Reinstall CryoID"):
                    bundled_blast_bin_dir()

    def test_bundled_payload_matches_pinned_manifest(self) -> None:
        import hashlib
        import json
        from D_BLAST_searching.blast_tools import BLAST_VERSION, BUNDLE_ROOT
        manifest = json.loads((BUNDLE_ROOT / "manifest.json").read_text())
        self.assertEqual(set(manifest), {"windows-x86_64", "linux-x86_64", "darwin-x86_64", "darwin-arm64"})
        for payload in manifest.values():
            self.assertEqual(payload["version"], BLAST_VERSION)
            for item in payload["files"]:
                content = (BUNDLE_ROOT / item["path"]).read_bytes()
                self.assertEqual(len(content), item["size"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), item["sha256"])

    def test_invalid_folder_reports_all_missing_programs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "blastp"):
                validate_blast_bin_dir(Path(tmp))

    def test_query_mapping_accepts_more_than_fifty_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            query_path = Path(tmp) / "queries.fasta"
            query_path.write_text(
                "".join(f">query_{index}\nGPLKYW\n" for index in range(51)),
                encoding="ascii",
            )
            params = Namespace(
                queryfile=str(query_path),
                reverse=True,
                logger=logging.getLogger("test.query_count"),
            )

            query_info = QueryMap(params)

            self.assertEqual(len(query_info), 51)


if __name__ == "__main__":
    unittest.main()
