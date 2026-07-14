import json
import shutil
import subprocess
import unittest
from pathlib import Path

from hand_landmarker.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _normalized(value):
    return str(value).replace("\\", "/")


class StageConfigRouteTests(unittest.TestCase):
    route_names = ("eval_val", "eval_test", "export", "infer")

    def test_default_configs_are_pretrain_routes(self):
        for name in self.route_names:
            with self.subTest(config=name):
                config = load_config(CONFIGS / "{}.yaml".format(name))
                self.assertEqual(config["model"]["checkpoint_stage"], "pretrain")
                self.assertIn(
                    "/hand_landmarker_runs/v1-pretrain-geometry/pretrain/checkpoints/best.weights.h5",
                    _normalized(config["hand"]["model_path"]),
                )

        self.assertIn(
            "/eval/pretrain/val",
            _normalized(load_config(CONFIGS / "eval_val.yaml")["output"]["dir"]),
        )
        self.assertIn(
            "/eval/pretrain/test",
            _normalized(load_config(CONFIGS / "eval_test.yaml")["output"]["dir"]),
        )
        self.assertIn(
            "/export/pretrain/",
            _normalized(load_config(CONFIGS / "export.yaml")["export"]["model_path"]),
        )
        self.assertTrue(
            _normalized(load_config(CONFIGS / "infer.yaml")["output"]["dir"]).endswith(
                "/hand_landmarker_inference/v1-pretrain-geometry"
            )
        )

    def test_train_configs_declare_checkpoint_stage(self):
        for stage in ("pretrain", "finetune"):
            with self.subTest(stage=stage):
                config = load_config(CONFIGS / "train_{}.yaml".format(stage))
                self.assertEqual(config["stage"], stage)
                self.assertEqual(config["model"]["checkpoint_stage"], stage)

    def test_pretrain_training_has_no_finetune_dependency(self):
        config = load_config(CONFIGS / "train_pretrain.yaml")
        self.assertEqual("pretrain", config["stage"])
        self.assertIsNone(config["training"]["initial_checkpoint"])
        serialized = json.dumps(config, ensure_ascii=False).replace("\\", "/").lower()
        self.assertNotIn("train_finetune_merged", serialized)
        self.assertNotIn("/v1/finetune/checkpoints/", serialized)

    def test_pretrain_curation_output_is_the_training_input(self):
        curation = load_config(CONFIGS / "curate_pretrain.yaml")
        training = load_config(CONFIGS / "train_pretrain.yaml")
        smoke = load_config(CONFIGS / "train_pretrain_smoke.yaml")
        curated_root = _normalized(curation["output"]["dir"])
        labels = _normalized(training["data"]["labels"])
        smoke_labels = _normalized(smoke["data"]["labels"])
        self.assertTrue(labels.startswith(curated_root + "/"))
        self.assertTrue(smoke_labels.startswith(curated_root + "/"))
        self.assertTrue(labels.endswith("hand_training_labels_pretrain_landmarks.jsonl"))
        self.assertTrue(smoke_labels.endswith("hand_training_labels_pretrain_smoke.jsonl"))
        fractions = training["sampling"]["sample_type_fractions"]
        self.assertEqual(0.0, fractions["NEG_RUNTIME_CANDIDATE"])
        self.assertEqual(0.0, fractions["NEG_LOW_PALM_CANDIDATE"])

    def test_conversion_dataset_sources_follow_export_stage(self):
        for stage in ("pretrain", "finetune"):
            with self.subTest(stage=stage):
                config = load_config(CONFIGS / "export_{}.yaml".format(stage))
                conversion = config["export"]["conversion_datasets"]
                self.assertTrue(conversion["enabled"])
                self.assertIn(
                    "/export/{}/model_conversion".format(stage),
                    _normalized(conversion["output_dir"]),
                )
                sources = conversion["sets"]
                self.assertEqual(
                    "configs/train_{}.yaml".format(stage),
                    sources["calibrate_datasets"]["sources"]["train"]["config_path"],
                )
                self.assertEqual(
                    "configs/eval_val_{}.yaml".format(stage),
                    sources["evaluate_datasets"]["sources"]["val"]["config_path"],
                )
                self.assertEqual(
                    "configs/eval_test_{}.yaml".format(stage),
                    sources["evaluate_datasets"]["sources"]["test"]["config_path"],
                )

    def test_stage_wrappers_extend_defaults_and_isolate_artifacts(self):
        outputs = {}
        for stage in ("pretrain", "finetune"):
            for name in self.route_names:
                with self.subTest(stage=stage, config=name):
                    path = CONFIGS / "{}_{}.yaml".format(name, stage)
                    self.assertTrue(
                        path.read_text(encoding="utf-8").startswith(
                            "extends: {}.yaml\n".format(name)
                        )
                    )
                    config = load_config(path)
                    self.assertEqual(config["model"]["checkpoint_stage"], stage)
                    expected_checkpoint = (
                        "/hand_landmarker_runs/v1-pretrain-geometry/pretrain/checkpoints/best.weights.h5"
                        if stage == "pretrain"
                        else "/hand_landmarker_runs/v1/finetune/checkpoints/best.weights.h5"
                    )
                    self.assertIn(expected_checkpoint, _normalized(config["hand"]["model_path"]))
                    if name.startswith("eval_"):
                        self.assertNotIn("palm", config)
                        self.assertEqual("roi", config["evaluation"]["mode"])
                        self.assertIn("hand_flag_threshold", config["evaluation"])
                        output = config["output"]["dir"]
                    elif name == "export":
                        output = config["export"]["model_path"]
                    else:
                        output = config["output"]["dir"]
                    outputs[(name, stage)] = _normalized(output)

        for name in self.route_names:
            self.assertNotEqual(outputs[(name, "pretrain")], outputs[(name, "finetune")])

        self.assertIn("/eval/pretrain/val", outputs[("eval_val", "pretrain")])
        self.assertIn("/eval/finetune/val", outputs[("eval_val", "finetune")])
        self.assertIn("/eval/pretrain/test", outputs[("eval_test", "pretrain")])
        self.assertIn("/eval/finetune/test", outputs[("eval_test", "finetune")])
        self.assertIn("/export/pretrain/", outputs[("export", "pretrain")])
        self.assertIn("/export/finetune/", outputs[("export", "finetune")])
        self.assertTrue(
            outputs[("infer", "pretrain")].endswith(
                "/hand_landmarker_inference/v1-pretrain-geometry"
            )
        )
        self.assertTrue(outputs[("infer", "finetune")].endswith("/inference/output/finetune"))


@unittest.skipUnless(shutil.which("make"), "GNU Make is required for route dry-run tests")
class MakeStageRouteTests(unittest.TestCase):
    make = shutil.which("make")

    def _dry_run(self, target, stage=None):
        command = [self.make, "-n"]
        if stage is not None:
            command.append("MODEL_STAGE={}".format(stage))
        command.append(target)
        return subprocess.run(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )

    def test_generic_targets_default_to_pretrain(self):
        expected = {
            "inspect": "configs/train_pretrain.yaml",
            "train": "configs/train_pretrain.yaml",
            "eval-val": "configs/eval_val_pretrain.yaml",
            "eval-test": "configs/eval_test_pretrain.yaml",
            "infer": "configs/infer_pretrain.yaml",
            "export": "configs/export_pretrain.yaml",
            "conversion-datasets": "configs/export_pretrain.yaml",
        }
        for target, config_path in expected.items():
            with self.subTest(target=target):
                result = self._dry_run(target)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn(config_path, result.stdout)
                self.assertNotIn("finetune", result.stdout.lower())

    def test_generic_targets_can_route_to_finetune(self):
        expected = {
            "inspect": "configs/train_finetune.yaml",
            "train": "configs/train_finetune.yaml",
            "eval-val": "configs/eval_val_finetune.yaml",
            "eval-test": "configs/eval_test_finetune.yaml",
            "infer": "configs/infer_finetune.yaml",
            "export": "configs/export_finetune.yaml",
            "conversion-datasets": "configs/export_finetune.yaml",
        }
        for target, config_path in expected.items():
            with self.subTest(target=target):
                result = self._dry_run(target, stage="finetune")
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn(config_path, result.stdout)

    def test_all_targets_include_both_stages_in_order(self):
        for target in ("inspect-all", "train-all"):
            with self.subTest(target=target):
                result = self._dry_run(target)
                self.assertEqual(result.returncode, 0, result.stdout)
                pretrain = result.stdout.find("configs/train_pretrain.yaml")
                finetune = result.stdout.find("configs/train_finetune.yaml")
                self.assertGreaterEqual(pretrain, 0, result.stdout)
                self.assertGreater(finetune, pretrain, result.stdout)

    def test_short_training_aliases_are_stage_explicit(self):
        for target, config_path in (
            ("pretrain", "configs/train_pretrain.yaml"),
            ("finetune", "configs/train_finetune.yaml"),
        ):
            with self.subTest(target=target):
                result = self._dry_run(target)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn(config_path, result.stdout)

    def test_invalid_model_stage_fails_closed(self):
        for stage in ("invalid", "pretrain finetune"):
            with self.subTest(stage=stage):
                result = self._dry_run("train", stage=stage)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("MODEL_STAGE must be", result.stdout)


if __name__ == "__main__":
    unittest.main()
