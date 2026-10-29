from __future__ import annotations

import subprocess
import os
import sys
import tempfile
import unittest
from pathlib import Path

from cryoid_pipeline.cli import decode_event
from cryoid_pipeline.models import pipeline_result_from_dict
from cryoid_pipeline.user_config import save_config
from cryoid_pipeline.user_config import load_config


class CryoIDCliTests(unittest.TestCase):
    def test_retired_options_are_rejected_without_modifying_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            save_config({"weights_dir": "keep-me"}, config)
            before = config.read_bytes()
            for key, value in (("blast-bin-dir", "missing"), ("rna-structure-cutoff-percent", "10"),
                               ("repo-root", "missing"), ("output-root", "missing"),
                               ("test-query-fasta", "missing.fasta")):
                commands = [
                    ["run", "--map", "map.mrc", "--pool", "pool.fasta", "--config", str(config), f"--{key}", value],
                    ["config", "--config", str(config), "set", key, value],
                    ["config", "--config", str(config), "unset", key],
                ]
                for command in commands:
                    with self.subTest(command=command):
                        result = subprocess.run([sys.executable, "-m", "cryoid_pipeline", *command],
                                                capture_output=True, text=True, check=False)
                        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                        self.assertEqual(config.read_bytes(), before)

    def test_help_and_config_show_only_current_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text('[defaults]\nblast_bin_dir = "missing"\n'
                              'rna_structure_cutoff_percent = "10"\n')
            for command in (["run", "--help"], ["config", "--config", str(config), "show"]):
                result = subprocess.run([sys.executable, "-m", "cryoid_pipeline", *command],
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                for retired in ("blast-bin-dir", "blast_bin_dir", "rna-structure-cutoff-percent",
                                "rna_structure_cutoff_percent", "repo-root", "output-root",
                                "test-query-fasta", "archived"):
                    self.assertNotIn(retired, result.stdout)

    def test_config_commands_set_show_unset_and_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            base_command = [
                sys.executable,
                "-m",
                "cryoid_pipeline",
                "config",
                "--config",
                str(config_path),
            ]

            path_result = subprocess.run(
                [*base_command, "path"], capture_output=True, text=True, encoding="utf-8", check=False
            )
            self.assertEqual(path_result.returncode, 0, path_result.stderr)
            self.assertEqual(Path(path_result.stdout.strip()), config_path.resolve())
            self.assertFalse(config_path.exists())

            set_result = subprocess.run(
                [*base_command, "set", "weights-dir", "saved-weights"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            self.assertEqual(set_result.returncode, 0, set_result.stderr)
            self.assertEqual(load_config(config_path)["weights_dir"], str(Path("saved-weights").resolve()))

            rcl_result = subprocess.run(
                [*base_command, "set", "rcl", "0.25"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            self.assertEqual(rcl_result.returncode, 0, rcl_result.stderr)
            self.assertEqual(load_config(config_path)["rcl"], "0.25")

            for key, value, stored_key in (
                ("mode", "rna", "sequence_mode"),
                ("protein-min-trace-length", "12", "protein_min_trace_length"),
                ("structure-selection", "0.2", "structure_selection"),
                ("keep-tmp", "true", "keep_tmp"),
                ("device", "auto", "device"),
                ("cpu-threads", "4", "cpu_threads"),
            ):
                setting_result = subprocess.run(
                    [*base_command, "set", key, value],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                )
                self.assertEqual(setting_result.returncode, 0, setting_result.stderr)
                self.assertEqual(load_config(config_path)[stored_key], value)

            show_result = subprocess.run(
                [*base_command, "show"], capture_output=True, text=True, encoding="utf-8", check=False
            )
            self.assertEqual(show_result.returncode, 0, show_result.stderr)
            self.assertIn(f"weights-dir = {Path('saved-weights').resolve()}", show_result.stdout)
            self.assertIn("rcl = 0.25", show_result.stdout)
            self.assertIn("mode = rna", show_result.stdout)
            self.assertIn("protein-min-trace-length = 12", show_result.stdout)
            self.assertIn("structure-selection = 0.2", show_result.stdout)
            self.assertIn("keep-tmp = true", show_result.stdout)
            self.assertNotIn("blast-bin-dir", show_result.stdout)
            self.assertIn("Exists: yes", show_result.stdout)

            unset_result = subprocess.run(
                [*base_command, "unset", "weights-dir"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            self.assertEqual(unset_result.returncode, 0, unset_result.stderr)
            self.assertEqual(load_config(config_path)["weights_dir"], "")

    def test_run_uses_current_directory_instead_of_saved_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "configured-project"
            config_path = root / "config.toml"
            save_config({"project_dir": str(project_dir)}, config_path)
            command = [
                sys.executable,
                "-m",
                "cryoid_pipeline",
                "run",
                "--map", str(root / "missing.mrc"),
                "--pool", str(root / "missing.fasta"),
                "--config",
                str(config_path),
                "--event-stream",
            ]
            env = os.environ.copy()
            env.pop("CRYOID_PROJECT_DIR", None)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )

            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            events = [event for line in completed.stdout.splitlines() if (event := decode_event(line))]
            result_event = next(event for event in events if event.get("type") == "result")
            result = pipeline_result_from_dict(result_event["result"])
            self.assertEqual(
                result.run_dir.parent,
                root.resolve() / "cryoid_results" / "runs",
            )

    def test_subprocess_reports_blocked_result_and_logs_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project"
            command = [
                sys.executable,
                "-m",
                "cryoid_pipeline",
                "run",
                "--map", str(root / "missing.mrc"),
                "--pool", str(root / "missing.fasta"),
                "--project",
                str(project_dir),
                "--event-stream",
            ]

            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )

            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            events = [event for line in completed.stdout.splitlines() if (event := decode_event(line))]
            result_events = [event for event in events if event.get("type") == "result"]
            self.assertEqual(len(result_events), 1)
            result = pipeline_result_from_dict(result_events[0]["result"])
            self.assertEqual(result.stages[0].name, "Input validation")
            self.assertEqual(result.stages[0].status, "blocked_missing_input")
            self.assertTrue((result.run_dir / "run_config.json").is_file())
            log_path = result.run_dir / "pipeline.log"
            self.assertTrue(log_path.is_file())
            self.assertIn("Command:", completed.stdout)
            self.assertIn("Command:", log_path.read_text(encoding="utf-8"))
            self.assertTrue(any(event.get("type") == "progress" for event in events))

    def test_preexisting_cancel_file_requests_graceful_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project"
            density_map = root / "input.mrc"
            candidate_pool = root / "pool.fasta"
            cancel_file = root / "cancel.signal"
            density_map.write_bytes(b"validation only")
            candidate_pool.write_text(">candidate\nACDEFGHIK\n", encoding="utf-8")
            cancel_file.touch()
            command = [
                sys.executable,
                "-m",
                "cryoid_pipeline",
                "run",
                "--map", str(density_map),
                "--pool", str(candidate_pool),
                "--project",
                str(project_dir),
                "--event-stream",
                "--cancel-file",
                str(cancel_file),
            ]

            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )

            self.assertEqual(completed.returncode, 130, completed.stdout + completed.stderr)
            events = [event for line in completed.stdout.splitlines() if (event := decode_event(line))]
            result_event = next(event for event in events if event.get("type") == "result")
            result = pipeline_result_from_dict(result_event["result"])
            statuses = {stage.name: stage.status for stage in result.stages}
            self.assertEqual(statuses["Run"], "skipped")
            self.assertFalse((result.run_dir / "tmp").exists())
            self.assertFalse(cancel_file.exists())


if __name__ == "__main__":
    unittest.main()
