from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cryoid_pipeline.user_config import (
    default_config_path,
    load_config,
    resolve_device,
    resolve_project_dir,
    resolve_protein_min_trace_length,
    resolve_rcl,
    resolve_sequence_mode,
    resolve_structure_selection,
    resolve_weights_dir,
    save_config,
    validate_structure_selection,
)
from cryoid_pipeline.structure_selection import resolve_effective_structure_selection


class UserConfigTests(unittest.TestCase):
    def test_standard_windows_and_linux_locations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            with patch("cryoid_pipeline.user_config.platform.system", return_value="Windows"):
                with patch.dict(os.environ, {"APPDATA": str(base)}, clear=True):
                    self.assertEqual(default_config_path(), base / "cryoid" / "config.toml")
            with patch("cryoid_pipeline.user_config.platform.system", return_value="Linux"):
                with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(base)}, clear=True):
                    self.assertEqual(default_config_path(), base / "cryoid" / "config.toml")
            with patch("cryoid_pipeline.user_config.platform.system", return_value="Darwin"):
                with patch("cryoid_pipeline.user_config.Path.home", return_value=base):
                    with patch.dict(os.environ, {}, clear=True):
                        self.assertEqual(default_config_path(),
                                         base / "Library" / "Application Support" / "cryoid" / "config.toml")

    def test_toml_round_trip_preserves_windows_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            expected = r"C:\Users\xinye\cryoid_weights\train_val_data"
            save_config(
                {
                    "weights_dir": expected,
                    "device": "cuda:0",
                    "rcl": "0.25",
                    "protein_min_trace_length": "12",
                    "structure_selection": "1.0",
                    "sequence_ranking_run_dir": r"C:\runs\run_2",
                },
                path,
            )

            loaded = load_config(path)

            self.assertEqual(loaded["weights_dir"], expected)
            self.assertEqual(loaded["device"], "cuda:0")
            self.assertEqual(loaded["rcl"], "0.25")
            self.assertEqual(loaded["protein_min_trace_length"], "12")
            self.assertEqual(loaded["structure_selection"], "1.0")
            self.assertEqual(loaded["sequence_ranking_run_dir"], r"C:\runs\run_2")
            text = path.read_text(encoding="utf-8")
            self.assertIn("[defaults]", text)
            self.assertIn("[gui]", text)

    def test_resolution_priority_is_argument_environment_config_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config = {
                "project_dir": str(root / "config-project"),
                "weights_dir": str(root / "config-weights"),
                "device": "cpu",
                "rcl": "0.1",
                "protein_min_trace_length": "11",
                "structure_selection": "0.25",
            }
            env = {
                "CRYOID_PROJECT_DIR": str(root / "env-project"),
                "CRYOID_UNET_WEIGHTS_DIR": str(root / "env-weights"),
                "CRYOID_UNET_DEVICE": "cuda:1",
                "CRYOID_RCL": "0.2",
                "CRYOID_PROTEIN_MIN_TRACE_LENGTH": "13",
                "CRYOID_STRUCTURE_SELECTION": "20",
            }
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(
                    resolve_project_dir(root / "arg-project", config),
                    root / "arg-project" / "cryoid_results",
                )
                self.assertEqual(
                    resolve_project_dir(None, config),
                    root / "env-project" / "cryoid_results",
                )
                self.assertEqual(resolve_weights_dir(root / "arg-weights", config), root / "arg-weights")
                self.assertEqual(resolve_weights_dir(None, config), root / "env-weights")
                self.assertEqual(resolve_device("cuda:2", config), "cuda:2")
                self.assertEqual(resolve_device(None, config), "cuda:1")
                self.assertEqual(resolve_rcl(0.3, config), 0.3)
                self.assertEqual(resolve_rcl(None, config), 0.2)
                self.assertEqual(resolve_protein_min_trace_length(None, config), 13)
                self.assertEqual(resolve_structure_selection(None, config), "20")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    resolve_project_dir(None, config),
                    root / "config-project" / "cryoid_results",
                )
                self.assertEqual(resolve_weights_dir(None, config), root / "config-weights")
                self.assertEqual(resolve_device(None, config), "cpu")
                self.assertEqual(resolve_rcl(None, config), 0.1)
                self.assertEqual(resolve_protein_min_trace_length(None, config), 11)
                self.assertEqual(resolve_structure_selection(None, config), "0.25")
                self.assertEqual(
                    resolve_project_dir(None, {}, default=root),
                    root / "cryoid_results",
                )

    def test_project_builtin_default_is_cryoid_results_under_current_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            current = Path(tmp).resolve()
            with patch.dict(os.environ, {}, clear=True), patch("pathlib.Path.cwd", return_value=current):
                self.assertEqual(resolve_project_dir(None, {}), current / "cryoid_results")
                self.assertEqual(
                    resolve_project_dir(current / "cryoid_results", {}),
                    current / "cryoid_results",
                )

    def test_sequence_mode_and_structure_selection_defaults_and_validation(self) -> None:
        self.assertEqual(resolve_protein_min_trace_length(None, {}), 10)
        self.assertEqual(resolve_protein_min_trace_length(17, {}), 17)
        with self.assertRaises(ValueError):
            resolve_protein_min_trace_length(0, {})
        with self.assertRaises(ValueError):
            resolve_protein_min_trace_length("10.5", {})
        self.assertEqual(resolve_sequence_mode(None, {}), "auto")
        self.assertEqual(resolve_sequence_mode("rna", {}), "rna")
        self.assertIsNone(resolve_structure_selection(None, {}))
        self.assertEqual(resolve_effective_structure_selection(None, "protein").raw, "1")
        self.assertEqual(resolve_effective_structure_selection(None, "rna").raw, "0.5")
        self.assertEqual(validate_structure_selection("1"), "1")
        self.assertEqual(validate_structure_selection("1.0"), "1.0")
        self.assertEqual(validate_structure_selection("0.2"), "0.2")
        with self.assertRaises(ValueError):
            validate_structure_selection("2.5")
        with self.assertRaises(ValueError):
            validate_structure_selection("0")

    def test_retired_configuration_is_not_loaded_or_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text('[defaults]\nblast_bin_dir = "missing"\n'
                                   'rna_structure_cutoff_percent = "10"\nweights_dir = "keep-me"\n')
            config = load_config(config_path)
            self.assertNotIn("blast_bin_dir", config)
            self.assertNotIn("rna_structure_cutoff_percent", config)
            self.assertIsNone(resolve_structure_selection(None, config))
            save_config(config, config_path)
            self.assertNotIn("blast_bin_dir", config_path.read_text())
            self.assertNotIn("rna_structure_cutoff_percent", config_path.read_text())
            self.assertEqual(load_config(config_path)["weights_dir"], str(Path(tmp) / "keep-me"))

    def test_only_canonical_sections_supply_configured_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text('weights_dir = "flat-value"\nrcl = "0.7"\n'
                                   '[defaults]\nproject_dir = "current-project"\nchimerax_path = "wrong-section"\n'
                                   '[gui]\nweights_dir = "wrong-section"\nlast_run_dir = "current-run"\n')
            config = load_config(config_path)
            self.assertEqual(config["weights_dir"], "")
            self.assertEqual(config["rcl"], "0")
            self.assertEqual(config["chimerax_path"], "")
            self.assertEqual(config["project_dir"], str(Path(tmp) / "current-project"))
            self.assertEqual(config["last_run_dir"], str(Path(tmp) / "current-run"))

    def test_runtime_config_uses_selected_settings_without_default_file_leakage(self) -> None:
        from cryoid_pipeline.runtime_config import RuntimeConfig

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            save_config({"weights_dir": str(Path(tmp) / "default-weights"), "device": "cpu"}, config_path)
            with patch.dict(os.environ, {"CRYOID_CONFIG": str(config_path)}, clear=True):
                regular = RuntimeConfig.from_env()
                selected = RuntimeConfig.from_env(user_config={}, unet_device="cpu")
                explicit = RuntimeConfig.from_env(user_config={"weights_dir": str(Path(tmp) / "selected")}, unet_device="cpu")
            self.assertEqual(regular.unet_device, "cpu")
            self.assertEqual(regular.unet_weights_dir, Path(tmp) / "default-weights")
            self.assertIsNone(selected.unet_weights_dir)
            self.assertEqual(selected.unet_device, "cpu")
            self.assertEqual(explicit.unet_weights_dir, Path(tmp) / "selected")


if __name__ == "__main__":
    unittest.main()
