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


class PretrainConfigRouteTests(unittest.TestCase):
    expected_configs = {
        "curate_pretrain.yaml",
        "train_geometry.yaml",
        "train_smoke.yaml",
        "train_multitask.yaml",
        "eval_val.yaml",
        "eval_test.yaml",
        "infer.yaml",
        "export.yaml",
    }

    def test_config_surface_is_small_and_pretrain_only(self):
        self.assertEqual(
            self.expected_configs,
            {path.name for path in CONFIGS.glob("*.yaml")},
        )
        for path in CONFIGS.glob("*.yaml"):
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("finetune", text, path.name)

    def test_every_model_consumer_routes_to_v2(self):
        names = (
            "train_geometry",
            "train_smoke",
            "train_multitask",
            "eval_val",
            "eval_test",
            "infer",
            "export",
        )
        for name in names:
            with self.subTest(config=name):
                config = load_config(CONFIGS / (name + ".yaml"))
                self.assertEqual("v2", config["model"]["version"])
                self.assertEqual("pretrain", config["model"]["checkpoint_stage"])

    def test_geometry_and_multitask_are_explicit_persisted_phases(self):
        curation = load_config(CONFIGS / "curate_pretrain.yaml")
        geometry = load_config(CONFIGS / "train_geometry.yaml")
        smoke = load_config(CONFIGS / "train_smoke.yaml")
        multitask = load_config(CONFIGS / "train_multitask.yaml")
        curated_root = _normalized(curation["output"]["dir"])
        self.assertTrue(_normalized(geometry["data"]["labels"]).startswith(curated_root + "/"))
        self.assertTrue(_normalized(smoke["data"]["labels"]).startswith(curated_root + "/"))
        self.assertTrue(_normalized(multitask["data"]["labels"]).startswith(curated_root + "/"))
        self.assertTrue(geometry["data"]["labels"].endswith("pretrain_landmarks.jsonl"))
        self.assertTrue(multitask["data"]["labels"].endswith("pretrain_multitask.jsonl"))
        self.assertIn("/geometry/checkpoints/best.weights.h5", _normalized(multitask["training"]["initial_checkpoint"]))
        self.assertEqual(0.0, geometry["sampling"]["sample_type_fractions"]["NEG_RUNTIME_CANDIDATE"])
        self.assertGreater(multitask["sampling"]["sample_type_fractions"]["NEG_RUNTIME_CANDIDATE"], 0.0)
        self.assertGreaterEqual(multitask["multitask_gate"]["minimum_confirmed_negatives"], 1)
        self.assertEqual("val_multitask_score", multitask["training"]["checkpoint"]["monitor"])
        self.assertTrue(multitask["training"]["multitask_monitor"]["enabled"])

    def test_evaluation_inference_and_export_select_phase_from_make(self):
        for name in ("eval_val", "eval_test", "infer", "export"):
            config = load_config(CONFIGS / (name + ".yaml"))
            self.assertIn(
                "/v2-pretrain-r1/geometry/checkpoints/best.weights.h5",
                _normalized(config["hand"]["model_path"]),
            )
        export = load_config(CONFIGS / "export.yaml")
        self.assertNotIn("LeakyRelu", export["export"]["a1_allowed_operators"])
        self.assertIn("Relu", export["export"]["a1_allowed_operators"])
        sources = export["export"]["conversion_datasets"]["sets"]
        self.assertEqual(
            "configs/train_geometry.yaml",
            sources["calibrate_datasets"]["sources"]["train"]["config_path"],
        )
        self.assertEqual(
            "configs/eval_val.yaml",
            sources["evaluate_datasets"]["sources"]["val"]["config_path"],
        )
        self.assertEqual(
            "configs/eval_test.yaml",
            sources["evaluate_datasets"]["sources"]["test"]["config_path"],
        )

    def test_no_config_serializes_a_v1_model(self):
        for path in CONFIGS.glob("*.yaml"):
            config = load_config(path)
            serialized = json.dumps(config, ensure_ascii=False)
            self.assertNotIn('"version": "v1"', serialized)


@unittest.skipUnless(shutil.which("make"), "GNU Make is required")
class MakePretrainRouteTests(unittest.TestCase):
    make = shutil.which("make")

    def _dry_run(self, target):
        return subprocess.run(
            [self.make, "-n", target],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )

    def test_makefile_declares_reproducible_identity(self):
        text = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("HAND_DATA_ROOT := /root/autodl-tmp", text)
        self.assertIn("HAND_PRETRAIN_CURATED_ID := v2-pretrain-r1", text)
        self.assertIn("HAND_PRETRAIN_RUN_ID := v2-pretrain-r1", text)

    def test_compact_targets_route_to_compact_configs(self):
        expected = {
            "curate": "configs/curate_pretrain.yaml",
            "inspect": "configs/train_geometry.yaml",
            "smoke": "configs/train_smoke.yaml",
            "train": "configs/train_geometry.yaml",
            "multitask": "configs/train_multitask.yaml",
            "eval-val": "configs/eval_val.yaml",
            "eval-test": "configs/eval_test.yaml",
            "infer": "configs/infer.yaml",
            "export": "configs/export.yaml",
            "conversion-data": "configs/export.yaml",
        }
        for target, config in expected.items():
            with self.subTest(target=target):
                result = self._dry_run(target)
                self.assertEqual(0, result.returncode, result.stdout)
                self.assertIn(config, result.stdout)

    def test_multitask_runs_gate_before_training(self):
        result = self._dry_run("multitask")
        self.assertEqual(0, result.returncode, result.stdout)
        gate = result.stdout.find("check_multitask_data.py")
        train = result.stdout.find("scripts/train.py")
        self.assertGreaterEqual(gate, 0, result.stdout)
        self.assertGreater(train, gate, result.stdout)


if __name__ == "__main__":
    unittest.main()
