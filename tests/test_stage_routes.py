from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from hand_landmarker.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


class Hlml4PublicSurfaceTests(unittest.TestCase):
    def test_config_surface_is_minimal_and_semantically_separated(self) -> None:
        self.assertEqual(
            {"datasets.yaml", "training.yaml", "evaluation.yaml", "inference.yaml", "deploy.yaml"},
            {path.name for path in CONFIGS.glob("*.yaml")},
        )

    def test_three_training_profiles_keep_v2_model(self) -> None:
        for stage, checkpoint_stage in (
            ("geometry", "pretrain"),
            ("multitask", "pretrain"),
            ("multi_finetune", "finetune"),
        ):
            with self.subTest(stage=stage), mock.patch.dict(
                os.environ, {"HLML_STAGE": stage}, clear=False
            ):
                config = load_config(CONFIGS / "training.yaml")
                self.assertEqual(stage, config["resolved_profile"])
                self.assertEqual("v2", config["model"]["version"])
                self.assertEqual(checkpoint_stage, config["model"]["checkpoint_stage"])
                self.assertEqual(checkpoint_stage, config["stage"])
                self.assertEqual([2, 2, 3, 4, 4, 6, 6], config["model"]["num_iterations"])
                self.assertEqual("tqdm", config["training"]["progress_bar"])

    def test_multitask_and_finetune_initialization_chain(self) -> None:
        with mock.patch.dict(os.environ, {"HLML_STAGE": "multitask"}, clear=False):
            multitask = load_config(CONFIGS / "training.yaml")
        with mock.patch.dict(os.environ, {"HLML_STAGE": "multi_finetune"}, clear=False):
            finetune = load_config(CONFIGS / "training.yaml")
        self.assertIn("/geometry/checkpoints/best.weights.h5", multitask["training"]["initial_checkpoint"].replace("\\", "/"))
        self.assertIn("/multitask/checkpoints/best.weights.h5", finetune["training"]["initial_checkpoint"].replace("\\", "/"))
        self.assertEqual(0.55, finetune["training"]["gold_fraction"])

    def test_dataset_config_has_mandatory_replay_and_id_membership(self) -> None:
        config = load_config(CONFIGS / "datasets.yaml")
        stage = config["stages"]["multi_finetune"]
        self.assertEqual(0.55, stage["hard_fraction"])
        self.assertEqual(0.45, stage["replay_fraction"])
        self.assertGreater(stage["replay_fraction"], 0.0)
        self.assertIn("selection_id", stage["selections"][0])
        self.assertIn("negative_dataset_id", config["stages"]["multitask"]["negative_datasets"][0])

    def test_dataset_config_freezes_iris_1_1_membership_and_variants(self) -> None:
        config = load_config(CONFIGS / "datasets.yaml")
        for stage in ("geometry", "multitask", "multi_finetune"):
            self.assertEqual(
                {"FullEnhance0801", "FullEnhance0803", "FullEnhance0810"},
                {entry["dataset_id"] for entry in config["stages"][stage]["datasets"]},
            )
            self.assertEqual(
                {"eos_2.0-rtmpose-hcf0813-gate"},
                {entry["proposal_variant"] for entry in config["stages"][stage]["datasets"]},
            )
        self.assertEqual(
            {"eos_2.0-rtmpose-hcf0813-gate", "eos_2.0-rtmpose-gate"},
            {entry["proposal_variant"] for entry in config["evaluation"]["val"]},
        )
        self.assertEqual(
            ["eos_2.0-rtmpose-gate"],
            [entry["proposal_variant"] for entry in config["evaluation"]["test"]],
        )
        self.assertTrue(
            all(entry.get("capture_source_ids") for entry in config["evaluation"]["val"])
        )
        self.assertTrue(
            all(entry.get("capture_source_ids") for entry in config["evaluation"]["test"])
        )

    def test_val_and_test_profiles_are_fixed_roi_only(self) -> None:
        for action in ("val", "test"):
            with self.subTest(action=action), mock.patch.dict(
                os.environ, {"HLML_EVALUATION_SPLIT": action}, clear=False
            ):
                config = load_config(CONFIGS / "evaluation.yaml")
                self.assertEqual("roi", config["evaluation"]["mode"])
                self.assertNotIn("palm", config)
                self.assertIn("/{}.jsonl".format(action), config["data"]["labels"].replace("\\", "/"))
        with mock.patch.dict(os.environ, {"HLML_EVALUATION_SPLIT": "test"}, clear=False):
            test = load_config(CONFIGS / "evaluation.yaml")
        self.assertFalse(test["evaluation"]["tune_thresholds"])
        self.assertFalse(test["output"]["overwrite"])

    def test_export_retains_static_v2_a1_contract(self) -> None:
        config = load_config(CONFIGS / "deploy.yaml")
        self.assertEqual("v2", config["model"]["version"])
        self.assertEqual([1, 256, 256], config["model"]["input_shape"])
        self.assertTrue(config["export"]["strict_a1_operators"])
        self.assertFalse(config["export"]["dynamic_batch"])
        self.assertEqual(["landmarks", "hand_flag", "handedness"], config["export"]["output_names"])
        conversion = config["export"]["conversion_datasets"]
        self.assertTrue(conversion["enabled"])
        self.assertEqual(100, conversion["sets"]["calibrate_datasets"]["sources"]["train"]["count"])
        self.assertEqual(25, conversion["sets"]["evaluate_datasets"]["sources"]["val"]["count"])
        self.assertEqual(25, conversion["sets"]["evaluate_datasets"]["sources"]["test"]["count"])
        self.assertNotIn("evaluation", config)
        self.assertNotIn("palm", config)

    def test_inference_config_is_cascade_only(self) -> None:
        config = load_config(CONFIGS / "inference.yaml")
        self.assertEqual("infer_folder", config["task"])
        self.assertIn("palm", config)
        self.assertEqual("eos-2.0", config["palm"]["model_id"])
        self.assertEqual(384, config["palm"]["input_width"])
        self.assertEqual(224, config["palm"]["input_height"])
        self.assertEqual("palm_detector", config["palm"]["models_root"])
        self.assertNotIn("model_path", config["palm"])
        self.assertIn("hand_roi", config)
        self.assertNotIn("evaluation", config)
        self.assertNotIn("export", config)

    def test_makefile_exposes_only_v4_high_level_routes(self) -> None:
        text = (ROOT / "Makefile").read_text(encoding="utf-8")
        for target in (
            "data-audit:",
            "geometry:",
            "multitask:",
            "mine-hard:",
            "multi-finetune:",
            "val:",
            "freeze-winner:",
            "locked-test:",
            "export:",
            "acceptance-smoke:",
        ):
            self.assertIn(target, text)
        for obsolete in ("pretrain-curate:", "eval-test-geometry:", "finetune-train:"):
            self.assertNotIn(obsolete, text)

    @unittest.skipUnless(shutil.which("make"), "make is not installed")
    def test_make_high_level_routes_resolve(self) -> None:
        for target in ("geometry", "multitask", "mine-hard", "multi-finetune", "val", "freeze-winner", "locked-test", "export"):
            with self.subTest(target=target):
                result = subprocess.run(
                    ["make", "-n", target],
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                self.assertEqual(0, result.returncode, result.stdout)
                self.assertIn("scripts/hlml.py", result.stdout)

    def test_public_surface_has_no_test_to_mining_route(self) -> None:
        cli = (ROOT / "scripts" / "hlml.py").read_text(encoding="utf-8")
        self.assertIn('stage_paths(Path(args.train_root), args.snapshot_id, "multitask")', cli)
        self.assertNotIn('stage_paths(Path(args.train_root), args.snapshot_id, "test")', cli)


if __name__ == "__main__":
    unittest.main()
