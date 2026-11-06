from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mrcfile
import numpy as np
try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("torch is not installed in this test environment") from exc

from C_Unet_model_predicting.unet_inference import predict_location, run_unet_inference


class FakeUNET(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(
            (x.shape[0], self.out_channels, x.shape[2], x.shape[3], x.shape[4]),
            dtype=x.dtype,
            device=x.device,
        )
        if self.out_channels > 1:
            output[:, 1, :, :, :] = 1.0
        return output


class UnetInferenceTests(unittest.TestCase):
    def test_mps_device_event_reports_apple_gpu(self) -> None:
        events = []
        with (
            patch("C_Unet_model_predicting.unet_inference.validate_weights"),
            patch("C_Unet_model_predicting.unet_inference.select_device", return_value="mps"),
            patch("C_Unet_model_predicting.unet_inference.load_weights_metadata",
                  side_effect=RuntimeError("stop after device event")),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after device event"):
                run_unet_inference(Path("input.mrc"), Path("predictions"),
                                   weights_dir=Path("weights"), device="mps", status=events.append)
        self.assertEqual(events, [{"name": "U-Net prediction", "device": "mps",
                                   "kind": "inference_device", "device_name": "Apple GPU (Metal)"}])

    def _write_map(self, path: Path) -> None:
        data = np.arange(1, 65, dtype=np.float32).reshape(4, 4, 4)
        path.parent.mkdir(parents=True, exist_ok=True)
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(data)
            mrc.voxel_size = 1.0

    def _write_weights(self, weights_dir: Path, *, full_rna_channels: bool = False) -> None:
        weights_dir.mkdir(parents=True, exist_ok=True)
        channel_counts = (
            {"secondary_strctures": 5, "key_atoms": 4, "residue_types_6letter": 9}
            if full_rna_channels
            else {"secondary_strctures": 4, "key_atoms": 2, "residue_types_6letter": 7}
        )
        (weights_dir / "class_weight_for_training.txt").write_text(
            json.dumps({name: [1] * count for name, count in channel_counts.items()}),
            encoding="utf-8",
        )
        (weights_dir / "epoch_num_path.txt").write_text(
            json.dumps({"secondary_strctures": 1, "key_atoms": 1, "residue_types_6letter": 1}),
            encoding="utf-8",
        )
        for model_type in ("secondary_strctures", "key_atoms", "residue_types_6letter"):
            checkpoint_dir = weights_dir / "checkpoint" / model_type
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {}}, checkpoint_dir / f"{model_type}.epoch1.pth.tar")

    def test_run_unet_inference_writes_raw_prediction_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_path = root / "emd_0001.map"
            output_root = root / "predictions"
            tmp_root = root / "run" / "tmp" / "unet"
            weights_dir = root / "weights"
            self._write_map(map_path)
            self._write_weights(weights_dir)

            percents: list[int] = []
            events: list[dict] = []
            with patch("C_Unet_model_predicting.unet_inference.UNET", FakeUNET):
                result = run_unet_inference(
                    map_path,
                    output_root,
                    weights_dir=weights_dir,
                    device="cpu",
                    tmp_root=tmp_root,
                    progress_percent=percents.append,
                    status=events.append,
                )

            self.assertEqual(events[0]["kind"], "inference_device")
            self.assertEqual(events[0]["device"], "cpu")
            for index, model_type in enumerate(("secondary_strctures", "key_atoms", "residue_types_6letter"), 1):
                model_events = [e for e in events if e.get("model") == model_type]
                self.assertEqual(model_events[0]["kind"], "model_loading")
                self.assertEqual(model_events[0]["model_index"], index)
                self.assertEqual(model_events[-1]["kind"], "model_finished")
                counts = [e for e in model_events if e["kind"] == "model_progress"]
                self.assertEqual(counts[0]["completed"], 0)
                self.assertGreater(counts[-1]["total"], 0)
                self.assertEqual(counts[-1]["completed"], counts[-1]["total"])
            self.assertEqual(result.bundle.layout, "checkpoint_inference")
            self.assertEqual(percents[0], 5)
            self.assertEqual(percents[-1], 55)
            self.assertTrue(all(left <= right for left, right in zip(percents, percents[1:])))
            self.assertTrue(result.bundle.raw_key_atoms_1_path.is_file())
            self.assertEqual(len(result.bundle.secondary_structure_paths), 3)
            self.assertEqual(len(result.bundle.residue_type_paths), 6)
            for path in result.bundle.secondary_structure_paths:
                self.assertTrue(path.is_file())
            self.assertEqual(len(result.generated_files), 20)
            self.assertTrue(any("residue_types_6letter" in path.name for path in result.generated_files))
            self.assertFalse(
                any(
                    f"_residue_types_{idx}_" in path.name
                    for path in result.generated_files
                    for idx in range(1, 25)
                )
            )
            self.assertFalse((output_root / "unrefined").exists())
            self.assertTrue((tmp_root / "unrefined").is_dir())
            self.assertTrue(any((tmp_root / "unrefined") in path.parents for path in result.generated_files))

    def test_rna_mode_runs_all_dependent_checkpoints_but_exposes_only_rna_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_path = root / "emd_0002.map"
            output_root = root / "predictions"
            weights_dir = root / "weights"
            self._write_map(map_path)
            self._write_weights(weights_dir, full_rna_channels=True)
            messages: list[str] = []

            with patch("C_Unet_model_predicting.unet_inference.UNET", FakeUNET):
                result = run_unet_inference(
                    map_path,
                    output_root,
                    weights_dir=weights_dir,
                    device="cpu",
                    tmp_root=root / "tmp",
                    progress=messages.append,
                    mode="rna",
                )

            loaded = [
                message.split()[3]
                for message in messages
                if message.startswith("UNet inference: loading ") and " checkpoint " in message
            ]
            self.assertEqual(loaded, ["secondary_strctures", "key_atoms", "residue_types_6letter"])
            self.assertIsNone(result.bundle.raw_key_atoms_1_path)
            self.assertEqual(set(result.bundle.key_atom_channels), {2, 3})
            self.assertEqual(set(result.bundle.secondary_structure_channels), {4})
            self.assertEqual(set(result.bundle.residue_type_channels), {7, 8})
            self.assertEqual(len(result.bundle.output_files), 5)
            self.assertTrue(all(path.is_file() for path in result.bundle.output_files))

    def test_predict_location_honors_cancel_between_cubes(self) -> None:
        class Cancelled(RuntimeError):
            pass

        calls = 0

        def cancel_after_first_cube() -> None:
            nonlocal calls
            calls += 1
            if calls >= 3:
                raise Cancelled("cancelled")

        map_data = np.ones((1, 51, 51, 51), dtype=np.float32)

        with self.assertRaises(Cancelled):
            predict_location(
                map_data,
                FakeUNET(1, 2),
                2,
                "cpu",
                progress=lambda _message: None,
                should_cancel=cancel_after_first_cube,
            )

        self.assertGreaterEqual(calls, 3)


if __name__ == "__main__":
    unittest.main()
