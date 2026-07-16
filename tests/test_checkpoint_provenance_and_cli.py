import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from hand_landmarker.contracts import (
    validate_checkpoint_path_stage,
    validate_model_checkpoint_stage,
)
from hand_landmarker.evaluation import evaluate_from_config, evaluate_hand_rois
from hand_landmarker.export import _finetune_training_provenance, export_from_config
from hand_landmarker.training import train_from_config
from hand_landmarker.visualization import infer_folder_from_config
from scripts import evaluate as evaluate_script
from scripts import export_onnx as export_script
from scripts import infer_folder as infer_script


def _positive_row():
    return {
        "crop_id": "roi-1",
        "dataset_id": "unit",
        "source_image": "source.png",
        "crop_path": "roi.png",
        "_resolved_crop_path": "roi.png",
        "hand_presence": {"present": True},
        "handedness": {"label": "Right"},
        "landmarks_crop_norm": [
            {"id": index, "x": 0.25, "y": 0.75} for index in range(21)
        ],
    }


class CheckpointStageContractTests(unittest.TestCase):
    def test_finetune_export_records_training_manifest_and_initial_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "sha256_manifest.json"
            initial = root / "multitask" / "best.weights.h5"
            initial.parent.mkdir()
            manifest.write_text("{}\n", encoding="utf-8")
            initial.write_bytes(b"multitask-checkpoint")
            train_config_path = root / "train_finetune.yaml"
            train_config_path.write_text(
                json.dumps(
                    {
                        "task": "train",
                        "stage": "finetune",
                        "model": {"checkpoint_stage": "finetune"},
                        "data": {"curation_manifest": str(manifest)},
                        "training": {"initial_checkpoint": str(initial)},
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "export": {
                    "conversion_datasets": {
                        "sets": {
                            "calibrate_datasets": {
                                "sources": {
                                    "train": {"config_path": str(train_config_path)}
                                }
                            }
                        }
                    }
                }
            }

            provenance = _finetune_training_provenance(config)
            self.assertEqual("finetune_export_provenance_v1", provenance["schema_version"])
            self.assertEqual(str(manifest.resolve()), provenance["curation_manifest"]["path"])
            self.assertEqual(
                str(initial.resolve()),
                provenance["initial_multitask_checkpoint"]["path"],
            )
            for artifact in provenance.values():
                if isinstance(artifact, dict) and "sha256" in artifact:
                    self.assertEqual(64, len(artifact["sha256"]))

    def test_checkpoint_stage_is_a_required_strict_enum(self):
        self.assertEqual(
            "pretrain",
            validate_model_checkpoint_stage(
                {"model": {"checkpoint_stage": "pretrain"}}
            ),
        )
        self.assertEqual(
            "finetune",
            validate_model_checkpoint_stage(
                {"model": {"checkpoint_stage": "finetune"}}
            ),
        )
        for value in (None, "", "Pretrain", "teacher", 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "model.checkpoint_stage"):
                    validate_model_checkpoint_stage(
                        {"model": {"checkpoint_stage": value}}
                    )

    def test_checkpoint_path_rejects_only_explicit_opposite_stage(self):
        config = {"model": {"checkpoint_stage": "pretrain"}}
        self.assertEqual(
            "pretrain",
            validate_checkpoint_path_stage(config, "/models/custom/best.weights.h5"),
        )
        self.assertEqual(
            "pretrain",
            validate_checkpoint_path_stage(
                config, "/models/pretrain/checkpoints/best.weights.h5"
            ),
        )
        for value in (
            "/models/finetune/checkpoints/best.weights.h5",
            "/models/hand_landmarker_finetune.weights.h5",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "model.checkpoint_stage"):
                    validate_checkpoint_path_stage(config, value)

    def test_eval_export_and_infer_reject_opposite_stage_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            opposite_weights = root / "finetune" / "checkpoints" / "best.weights.h5"
            common_model = {"checkpoint_stage": "pretrain"}
            with self.assertRaisesRegex(ValueError, "model.checkpoint_stage"):
                evaluate_from_config(
                    {
                        "split": "val",
                        "model": common_model,
                        "hand": {"model_path": str(opposite_weights)},
                        "evaluation": {"mode": "roi"},
                        "output": {"dir": str(root / "eval-output")},
                    }
                )
            with self.assertRaisesRegex(ValueError, "model.checkpoint_stage"):
                export_from_config(
                    {
                        "model": common_model,
                        "hand": {"model_path": str(opposite_weights)},
                        "export": {"model_path": str(root / "model.onnx")},
                    }
                )
            with self.assertRaisesRegex(ValueError, "model.checkpoint_stage"):
                infer_folder_from_config(
                    {
                        "model": common_model,
                        "input": {"images_dir": str(input_dir)},
                        "hand": {"model_path": str(opposite_weights)},
                        "output": {"dir": str(root / "infer-output")},
                    }
                )

    def test_training_stage_must_match_model_checkpoint_stage(self):
        with self.assertRaisesRegex(ValueError, "model.checkpoint_stage"):
            train_from_config(
                {
                    "task": "train",
                    "stage": "pretrain",
                    "model": {"checkpoint_stage": "finetune"},
                }
            )

    def test_public_entries_reject_invalid_checkpoint_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "model.checkpoint_stage"):
                evaluate_from_config(
                    {
                        "split": "val",
                        "model": {"checkpoint_stage": "teacher"},
                        "evaluation": {"mode": "roi"},
                        "output": {"dir": directory},
                    }
                )
        with self.assertRaisesRegex(ValueError, "model.checkpoint_stage"):
            export_from_config({"model": {"checkpoint_stage": "teacher"}})
        with self.assertRaisesRegex(ValueError, "model.checkpoint_stage"):
            infer_folder_from_config({"model": {"checkpoint_stage": "teacher"}})

    @mock.patch(
        "hand_landmarker.evaluation.read_image",
        return_value=np.zeros((256, 256), dtype=np.uint8),
    )
    @mock.patch("hand_landmarker.evaluation.create_hand_predictor")
    def test_evaluation_summary_and_rows_record_checkpoint_stage(
        self, create_hand, _read_image
    ):
        hand = SimpleNamespace(
            landmarks_norm=[(0.25, 0.75)] * 21,
            hand_flag_score=0.9,
            handedness="Right",
            handedness_score=0.8,
        )
        predictor = mock.Mock()
        predictor.predict.return_value = [hand]
        create_hand.return_value = predictor
        with mock.patch.object(Path, "is_file", return_value=True):
            report = evaluate_hand_rois(
                {
                    "model": {"checkpoint_stage": "pretrain"},
                    "data": {},
                    "evaluation": {"tune_thresholds": False},
                },
                [_positive_row()],
            )
        self.assertEqual("pretrain", report["model_checkpoint_stage"])
        self.assertEqual(
            "pretrain", report["details"][0]["model_checkpoint_stage"]
        )

    @mock.patch("hand_landmarker.visualization.CascadeRunner")
    @mock.patch(
        "hand_landmarker.visualization.read_image",
        return_value=np.zeros((720, 1280), dtype=np.uint8),
    )
    def test_inference_summary_and_jsonl_rows_record_checkpoint_stage(
        self, _read_image, cascade_runner
    ):
        cascade_runner.return_value.predict.return_value = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "frame.png").write_bytes(b"unit-test")
            hand_model = root / "hand.weights.h5"
            palm_model = root / "palm.onnx"
            hand_model.write_bytes(b"hand")
            palm_model.write_bytes(b"palm")
            summary = infer_folder_from_config(
                {
                    "model": {"checkpoint_stage": "finetune"},
                    "input": {"images_dir": str(input_dir), "recursive": False},
                    "hand": {"model_path": str(hand_model)},
                    "palm": {"model_path": str(palm_model)},
                    "output": {
                        "dir": str(output_dir),
                        "write_annotated_images": False,
                    },
                }
            )
            rows = [
                json.loads(line)
                for line in (output_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
        self.assertEqual("finetune", summary["model_checkpoint_stage"])
        self.assertEqual(
            "finetune", summary["models"]["hand"]["checkpoint_stage"]
        )
        self.assertEqual("finetune", rows[0]["model_checkpoint_stage"])

    def test_export_contract_records_checkpoint_stage(self):
        class FakeDimension:
            def HasField(self, _name):
                return False

        def value_info(name, shape):
            tensor_type = SimpleNamespace(
                shape=SimpleNamespace(dim=[FakeDimension() for _ in shape]),
                elem_type=1,
            )
            return SimpleNamespace(
                name=name,
                expected_shape=shape,
                type=SimpleNamespace(tensor_type=tensor_type),
            )

        graph = SimpleNamespace(
            graph=SimpleNamespace(
                input=[value_info("inputs", [1, 1, 256, 256])],
                output=[
                    value_info("landmarks", [1, 42, 1, 1]),
                    value_info("hand_flag", [1, 1, 1, 1]),
                    value_info("handedness", [1, 1, 1, 1]),
                ],
                node=[],
                initializer=[],
            )
        )

        class FakeModel:
            input_shape = (None, 1, 256, 256)
            output_shape = [
                (None, 1, 1, 42),
                (None, 1, 1, 1),
                (None, 1, 1, 1),
            ]
            output_names = [
                "convld_21_2d",
                "activation_handflag",
                "activation_handedness",
            ]

            def load_weights(self, _path):
                return None

            def count_params(self):
                return 123

            def __call__(self, _tensor, training=False):
                del training
                value = float(np.mean(_tensor))
                return [
                    np.full((1, 1, 1, 42), value, dtype=np.float32),
                    np.full((1, 1, 1, 1), value, dtype=np.float32),
                    np.full((1, 1, 1, 1), value, dtype=np.float32),
                ]

        class FakeSession:
            def __init__(self, _path, providers=None):
                del providers

            def get_inputs(self):
                return [SimpleNamespace(name="inputs")]

            def run(self, _outputs, _feeds):
                value = float(np.mean(next(iter(_feeds.values()))))
                return [
                    np.full((1, 42, 1, 1), value, dtype=np.float32),
                    np.full((1, 1, 1, 1), value, dtype=np.float32),
                    np.full((1, 1, 1, 1), value, dtype=np.float32),
                ]

        def convert_from_keras(_model, input_signature, opset, output_path):
            del input_signature, opset
            Path(output_path).write_bytes(b"fake-onnx")

        fake_modules = {
            "onnx": SimpleNamespace(
                load=lambda _path: graph,
                checker=SimpleNamespace(check_model=lambda _graph: None),
                TensorProto=SimpleNamespace(FLOAT=1),
                helper=SimpleNamespace(get_attribute_value=lambda attribute: attribute),
            ),
            "onnxruntime": SimpleNamespace(InferenceSession=FakeSession),
            "tensorflow": SimpleNamespace(
                float32="float32",
                TensorSpec=lambda shape, dtype, name: (shape, dtype, name),
            ),
            "tf2onnx": SimpleNamespace(
                convert=SimpleNamespace(from_keras=convert_from_keras)
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "pretrain.weights.h5"
            output = root / "pretrain.onnx"
            weights.write_bytes(b"weights")
            with mock.patch.dict(sys.modules, fake_modules):
                with mock.patch(
                    "models.hand_landmarker.registry.build_model",
                    return_value=FakeModel(),
                ):
                    with mock.patch(
                        "models.hand_landmarker.registry.reparameterize_for_deploy",
                        side_effect=lambda model, **_kwargs: model,
                    ), mock.patch(
                        "hand_landmarker.export._shape_from_onnx",
                        side_effect=lambda item: item.expected_shape,
                    ):
                        with mock.patch(
                            "hand_landmarker.export.generate_conversion_datasets",
                            return_value={"status": "ok", "counts": {"calibrate_datasets": 20}},
                        ) as generate_conversion:
                            report = export_from_config(
                                {
                                    "model": {"checkpoint_stage": "pretrain"},
                                    "hand": {"model_path": str(weights)},
                                    "export": {
                                        "model_path": str(output),
                                        "conversion_datasets": {
                                            "enabled": True,
                                            "output_dir": str(root / "model_conversion"),
                                        },
                                    },
                                }
                            )
                            graph.graph.node = [
                                SimpleNamespace(
                                    name="runtime_transpose",
                                    op_type="Transpose",
                                    input=[],
                                    attribute=[],
                                )
                            ]
                            strict_output = root / "strict-transpose.onnx"
                            with self.assertRaisesRegex(ValueError, "Transpose"):
                                export_from_config(
                                    {
                                        "model": {"checkpoint_stage": "pretrain"},
                                        "hand": {"model_path": str(weights)},
                                        "export": {"model_path": str(strict_output)},
                                    }
                                )
                            self.assertFalse(strict_output.exists())

                            forced_output = root / "forced-transpose.onnx"
                            forced_report = export_from_config(
                                {
                                    "model": {"checkpoint_stage": "pretrain"},
                                    "hand": {"model_path": str(weights)},
                                    "export": {
                                        "model_path": str(forced_output),
                                        "force_a1_operator_export": True,
                                    },
                                }
                            )
                            self.assertEqual(2, generate_conversion.call_count)
            contract = json.loads(
                output.with_suffix(".contract.json").read_text(encoding="utf-8")
            )
            forced_contract = json.loads(
                forced_output.with_suffix(".contract.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual("pretrain", report["model_checkpoint_stage"])
        self.assertEqual("pretrain", contract["model_checkpoint_stage"])
        self.assertEqual("ok", contract["conversion_datasets"]["status"])
        self.assertFalse(contract["a1_operator_audit"]["forced"])
        self.assertTrue(contract["a1_operator_audit"]["enforced"])
        self.assertEqual(
            ["Transpose"], forced_report["a1_operator_audit"]["unsupported"]
        )
        self.assertTrue(forced_contract["a1_operator_audit"]["forced"])
        self.assertFalse(forced_contract["a1_operator_audit"]["enforced"])


class CliOverrideTests(unittest.TestCase):
    def test_evaluate_cli_overrides_model_output_and_overwrite(self):
        config = {
            "hand": {"model_path": "old.weights.h5"},
            "output": {"dir": "old-output", "overwrite": False},
        }
        evaluate_script._apply_cli_overrides(
            config,
            SimpleNamespace(
                model_path="new.weights.h5",
                output_dir="new-output",
                overwrite=True,
            ),
        )
        self.assertEqual("new.weights.h5", config["hand"]["model_path"])
        self.assertEqual("new-output", config["output"]["dir"])
        self.assertTrue(config["output"]["overwrite"])

    def test_export_cli_overrides_all_artifact_paths(self):
        config = {
            "hand": {"model_path": "old.weights.h5"},
            "export": {"model_path": "old.onnx", "overwrite": False},
        }
        export_script._apply_cli_overrides(
            config,
            SimpleNamespace(
                weights_path="new.weights.h5",
                output_path="new.onnx",
                contract_path="new.contract.json",
                conversion_output_dir="new-conversion-output",
                overwrite=True,
                force=False,
            ),
        )
        self.assertEqual("new.weights.h5", config["hand"]["model_path"])
        self.assertEqual("new.onnx", config["export"]["model_path"])
        self.assertEqual("new.contract.json", config["export"]["contract_path"])
        self.assertEqual(
            "new-conversion-output",
            config["export"]["conversion_datasets"]["output_dir"],
        )
        self.assertTrue(config["export"]["overwrite"])
        self.assertNotIn("force_a1_operator_export", config["export"])

    def test_export_cli_force_only_enables_a1_operator_bypass(self):
        config = {
            "export": {
                "strict_a1_operators": True,
                "overwrite": False,
                "validate": {"enabled": True},
            }
        }
        export_script._apply_cli_overrides(
            config,
            SimpleNamespace(
                weights_path=None,
                output_path=None,
                contract_path=None,
                conversion_output_dir=None,
                overwrite=False,
                force=True,
            ),
        )
        self.assertTrue(config["export"]["strict_a1_operators"])
        self.assertTrue(config["export"]["force_a1_operator_export"])
        self.assertFalse(config["export"]["overwrite"])
        self.assertEqual({"enabled": True}, config["export"]["validate"])

    def test_infer_cli_output_override_rehomes_jsonl(self):
        config = {
            "hand": {"model_path": "old.weights.h5"},
            "output": {
                "dir": "old-output",
                "overwrite": False,
            },
        }
        infer_script._apply_cli_overrides(
            config,
            SimpleNamespace(
                model_path="new.weights.h5",
                output_dir="new-output",
                overwrite=True,
            ),
        )
        self.assertEqual("new.weights.h5", config["hand"]["model_path"])
        self.assertEqual("new-output", config["output"]["dir"])
        self.assertTrue(config["output"]["overwrite"])


if __name__ == "__main__":
    unittest.main()
