import importlib.util
import unittest
from pathlib import Path

from models.hand_landmarker.registry import DEFAULT_VERSION, available_versions


ROOT = Path(__file__).resolve().parents[1]
HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None


class V2StaticContractTests(unittest.TestCase):
    def test_registry_exposes_only_v2(self):
        self.assertEqual("v2", DEFAULT_VERSION)
        self.assertEqual(("v2",), available_versions())

    def test_v2_source_has_no_leaky_relu_and_defines_fusion(self):
        text = (ROOT / "models" / "hand_landmarker" / "v2.py").read_text(encoding="utf-8")
        self.assertNotIn("LeakyReLU", text)
        self.assertIn("class RepConv2D", text)
        self.assertIn("class RepDepthwiseConv2D", text)
        self.assertIn("def reparameterize_for_deploy", text)


@unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow is available only in the server environment")
class V2TensorFlowContractTests(unittest.TestCase):
    def test_interface_and_branch_fusion_parity(self):
        import numpy as np

        from models.hand_landmarker.registry import build_model, reparameterize_for_deploy

        iterations = [1, 1, 1, 1, 1, 1, 1]
        training_model = build_model("v2", num_iterations=iterations)
        deploy_model = reparameterize_for_deploy(
            training_model, version="v2", num_iterations=iterations
        )
        self.assertEqual((None, 1, 256, 256), tuple(training_model.input_shape))
        self.assertEqual(
            [(None, 1, 1, 42), (None, 1, 1, 1), (None, 1, 1, 1)],
            [tuple(shape) for shape in training_model.output_shape],
        )
        self.assertEqual(
            ["convld_21_2d", "activation_handflag", "activation_handedness"],
            list(training_model.output_names),
        )
        self.assertLess(deploy_model.count_params(), training_model.count_params())
        tensor = np.random.RandomState(7).uniform(0, 1, (1, 1, 256, 256)).astype("float32")
        expected = training_model(tensor, training=False)
        actual = deploy_model(tensor, training=False)
        for left, right in zip(expected, actual):
            self.assertTrue(np.allclose(left, right, atol=5e-5, rtol=5e-4))


if __name__ == "__main__":
    unittest.main()
