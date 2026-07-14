import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from hand_landmarker.conversion_datasets import (
    ConversionDatasetError,
    deterministic_stratified_sample,
    generate_conversion_datasets,
    guard_conversion_dataset_output,
)
from hand_landmarker.io_utils import sha256_file, write_image


def _row(identity, image_path, dataset_id, sample_type="POS_RUNTIME", handedness="Right"):
    return {
        "global_crop_id": identity,
        "crop_id": identity,
        "dataset_id": dataset_id,
        "sample_type": sample_type,
        "hand_presence": {"present": True},
        "handedness": {"label": handedness},
        "_resolved_crop_path": str(image_path),
    }


class DeterministicSelectionTests(unittest.TestCase):
    def test_selection_is_stratified_and_independent_of_input_order(self):
        records = []
        for index in range(20):
            records.append(
                {
                    "global_crop_id": "record-{:02d}".format(index),
                    "sample_type": "positive" if index < 10 else "negative",
                }
            )
        first = deterministic_stratified_sample(records, 8, ["sample_type"], "unit")
        second = deterministic_stratified_sample(
            list(reversed(records)), 8, ["sample_type"], "unit"
        )
        self.assertEqual(
            [row["global_crop_id"] for row in first],
            [row["global_crop_id"] for row in second],
        )
        counts = {
            value: sum(row["sample_type"] == value for row in first)
            for value in ("positive", "negative")
        }
        self.assertEqual({"positive": 4, "negative": 4}, counts)

    def test_selection_fails_when_source_is_too_small(self):
        with self.assertRaisesRegex(ConversionDatasetError, "contains only"):
            deterministic_stratified_sample(
                [{"global_crop_id": "only-one"}], 2, [], "unit"
            )


class ConversionDatasetGenerationTests(unittest.TestCase):
    def _source_config(self, root, name, text):
        path = root / "{}.yaml".format(name)
        path.write_text(text, encoding="utf-8")
        return path

    def test_generation_is_read_only_reproducible_and_matches_strict_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images_dir = root / "canonical-rois"
            images_dir.mkdir()
            rows = {"train": [], "val": [], "test": []}
            source_counts = {"train": 20, "val": 5, "test": 5}
            cursor = 0
            for source, count in source_counts.items():
                for index in range(count):
                    path = images_dir / "{}-{:02d}.png".format(source, index)
                    write_image(path, np.full((256, 256), cursor % 256, dtype=np.uint8))
                    rows[source].append(
                        _row(
                            "{}-{:02d}".format(source, index),
                            path,
                            "dataset-{}".format(index % 2),
                            sample_type="type-{}".format(index % 2),
                            handedness="Left" if index % 2 else "Right",
                        )
                    )
                    cursor += 1

            train_config = self._source_config(
                root,
                "train",
                "task: train\nstage: pretrain\nmodel:\n  checkpoint_stage: pretrain\ndata: {}\n",
            )
            val_config = self._source_config(
                root,
                "val",
                "task: evaluate\nsplit: val\nmodel:\n  checkpoint_stage: pretrain\ndata: {}\n",
            )
            test_config = self._source_config(
                root,
                "test",
                "task: evaluate\nsplit: test\nmodel:\n  checkpoint_stage: pretrain\ndata: {}\n",
            )
            output_root = root / "export" / "pretrain" / "model_conversion"
            config = {
                "_meta": {"repo_root": str(root), "config_dir": str(root)},
                "model": {"checkpoint_stage": "pretrain"},
                "export": {
                    "overwrite": False,
                    "conversion_datasets": {
                        "enabled": True,
                        "output_dir": str(output_root),
                        "selection_salt": "unit-test",
                        "input": {
                            "shape": [1, 1, 256, 256],
                            "dtype": "float32",
                            "normalization": "uint8_div_255",
                        },
                        "sets": {
                            "calibrate_datasets": {
                                "minimum_count": 20,
                                "sources": {
                                    "train": {
                                        "config_path": str(train_config),
                                        "count": 20,
                                        "stratify_by": ["sample_type", "dataset_id"],
                                    }
                                },
                            },
                            "evaluate_datasets": {
                                "minimum_count": 10,
                                "sources": {
                                    "val": {
                                        "config_path": str(val_config),
                                        "count": 5,
                                        "stratify_by": ["dataset_id", "handedness.label"],
                                    },
                                    "test": {
                                        "config_path": str(test_config),
                                        "count": 5,
                                        "stratify_by": ["dataset_id", "handedness.label"],
                                    },
                                },
                            },
                        },
                    },
                },
            }
            original_hashes = {path: sha256_file(path) for path in images_dir.iterdir()}

            def audit(source_config, **_kwargs):
                if source_config.get("task") == "train":
                    key = "train"
                else:
                    key = source_config["split"]
                return [dict(row) for row in rows[key]], {
                    "labels": "{}.jsonl".format(key),
                    "labels_sha256": "{}-labels-sha".format(key),
                }

            with mock.patch(
                "hand_landmarker.conversion_datasets.audit_canonical_dataset",
                side_effect=audit,
            ):
                report = generate_conversion_datasets(config)
                first_archive_sha = report["archive_sha256"]
                config["export"]["overwrite"] = True
                for values in rows.values():
                    values.reverse()
                second_report = generate_conversion_datasets(config)

            self.assertEqual(first_archive_sha, second_report["archive_sha256"])
            self.assertEqual(
                {"calibrate_datasets": 20, "evaluate_datasets": 10},
                report["counts"],
            )
            self.assertEqual(
                original_hashes,
                {path: sha256_file(path) for path in images_dir.iterdir()},
            )

            datasets_dir = output_root / "datasets"
            self.assertEqual(
                {"calibrate_datasets", "evaluate_datasets"},
                {path.name for path in datasets_dir.iterdir() if path.is_dir()},
            )
            files = sorted(path for path in datasets_dir.rglob("*") if path.is_file())
            self.assertEqual(30, len(files))
            self.assertTrue(all(path.suffix == ".npy" for path in files))
            self.assertEqual([], [path for path in datasets_dir.rglob("*") if path.is_dir() and path.parent != datasets_dir])
            for path in files:
                tensor = np.load(path, allow_pickle=False)
                self.assertEqual((1, 1, 256, 256), tensor.shape)
                self.assertEqual(np.float32, tensor.dtype)
                self.assertGreaterEqual(float(tensor.min()), 0.0)
                self.assertLessEqual(float(tensor.max()), 1.0)

            expected_zip_entries = sorted(
                "datasets/{}".format(path.relative_to(datasets_dir).as_posix())
                for path in files
            )
            with zipfile.ZipFile(output_root / "datasets.zip") as archive:
                self.assertEqual(expected_zip_entries, sorted(archive.namelist()))
            self.assertFalse((datasets_dir / "datasets_manifest.json").exists())
            manifest = json.loads(
                (output_root / "datasets_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["input_contract"]["contains_model_outputs"])
            self.assertEqual(30, len(manifest["files"]))

    def test_preflight_guard_protects_existing_conversion_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model_conversion"
            output.mkdir()
            config = {
                "export": {
                    "overwrite": False,
                    "conversion_datasets": {
                        "enabled": True,
                        "output_dir": str(output),
                    },
                }
            }
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                guard_conversion_dataset_output(config)
            guard_conversion_dataset_output(config, overwrite=True)

    def test_output_must_not_contain_onnx_or_contract_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage-export"
            config = {
                "export": {
                    "model_path": str(output / "hand.onnx"),
                    "conversion_datasets": {
                        "enabled": True,
                        "output_dir": str(output),
                    },
                }
            }
            with self.assertRaisesRegex(ConversionDatasetError, "must be independent"):
                guard_conversion_dataset_output(config)


if __name__ == "__main__":
    unittest.main()
