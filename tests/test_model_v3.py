import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None


class V3StaticContractTests(unittest.TestCase):
    def test_v3_family_has_no_a1_unsupported_leaky_relu(self):
        for name in ("pro.py", "max.py", "lite.py"):
            text = (
                ROOT / "models" / "hand_landmarker" / "v3" / name
            ).read_text(encoding="utf-8")
            self.assertNotIn("LeakyReLU", text)

    def test_v3_max_declares_material_multibranch_training_graph(self):
        text = (
            ROOT / "models" / "hand_landmarker" / "v3" / "max.py"
        ).read_text(encoding="utf-8")
        self.assertIn("TRAIN_CONV_BRANCHES = 4", text)
        self.assertIn("class RepMultiBranchConv2D", text)
        self.assertIn("class RepMultiBranchDepthwiseConv2D", text)
        self.assertIn("scale_conv", text)
        self.assertIn("identity_bn", text)


@unittest.skipUnless(
    HAS_TENSORFLOW, "TensorFlow is available only in the server environment"
)
class V3TensorFlowContractTests(unittest.TestCase):
    def test_parameter_tiers_and_max_fusion_parity(self):
        import numpy as np

        from models.hand_landmarker.registry import (
            build_model,
            reparameterize_for_deploy,
        )
        from scripts.build_export_preflight import (
            prepare_quantization_probe_weights,
        )

        iterations = [1, 1, 1, 1, 1, 1, 1]
        training = {
            version: build_model(version, num_iterations=iterations)
            for version in ("v3-pro", "v3-max", "v3-lite")
        }
        deploy = {
            version: reparameterize_for_deploy(
                training[version],
                version=version,
                num_iterations=iterations,
            )
            for version in training
        }

        for model in training.values():
            self.assertEqual((None, 1, 256, 256), tuple(model.input_shape))
            self.assertEqual(
                [(None, 1, 1, 42), (None, 1, 1, 1), (None, 1, 1, 1)],
                [tuple(shape) for shape in model.output_shape],
            )
        self.assertGreater(
            training["v3-max"].count_params(),
            3 * training["v3-pro"].count_params(),
        )
        self.assertEqual(
            deploy["v3-pro"].count_params(),
            deploy["v3-max"].count_params(),
        )
        self.assertLess(
            deploy["v3-lite"].count_params(),
            0.7 * deploy["v3-pro"].count_params(),
        )

        prepare_quantization_probe_weights(training["v3-max"], seed=20260818)
        deploy_max = reparameterize_for_deploy(
            training["v3-max"],
            version="v3-max",
            num_iterations=iterations,
        )
        tensor = np.random.RandomState(13).uniform(
            0.0, 1.0, (1, 1, 256, 256)
        ).astype("float32")
        expected = training["v3-max"](tensor, training=False)
        actual = deploy_max(tensor, training=False)
        for left, right in zip(expected, actual):
            self.assertTrue(np.allclose(left, right, atol=5e-5, rtol=5e-4))


if __name__ == "__main__":
    unittest.main()
