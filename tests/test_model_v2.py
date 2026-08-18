import importlib.util
import unittest
from pathlib import Path

from models.hand_landmarker.registry import DEFAULT_VERSION, available_versions


ROOT = Path(__file__).resolve().parents[1]
HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None


class V2StaticContractTests(unittest.TestCase):
    def test_registry_exposes_v2_and_v3_family(self):
        self.assertEqual("v3-pro", DEFAULT_VERSION)
        self.assertEqual(("v2", "v3-lite", "v3-max", "v3-pro"), available_versions())

    def test_v2_source_has_no_leaky_relu_and_defines_fusion(self):
        text = (ROOT / "models" / "hand_landmarker" / "v2.py").read_text(encoding="utf-8")
        self.assertNotIn("LeakyReLU", text)
        self.assertIn("class RepConv2D", text)
        self.assertIn("class RepDepthwiseConv2D", text)
        self.assertIn("def reparameterize_for_deploy", text)
        self.assertNotIn("aux_conv", text)
        self.assertIn("zero_gamma=True", text)


@unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow is available only in the server environment")
class V2TensorFlowContractTests(unittest.TestCase):
    def test_preflight_quantization_probe_removes_training_stability_zeros(self):
        import numpy as np

        from models.hand_landmarker.registry import build_model
        from scripts.build_export_preflight import prepare_quantization_probe_weights

        model = build_model("v2", num_iterations=[1, 1, 1, 1, 1, 1, 1])
        report = prepare_quantization_probe_weights(model, seed=20260714)
        self.assertGreater(report["activated_batch_norm_count"], 0)
        self.assertEqual(
            ["convld_21_2d", "conv_handflag", "conv_handedness"],
            report["activated_heads"],
        )
        for name in report["activated_heads"]:
            kernel, _bias = model.get_layer(name).get_weights()
            self.assertTrue(np.any(kernel != 0.0))
        for layer in model.layers:
            batch_norm = getattr(layer, "train_bn", None)
            if batch_norm is not None:
                gamma = batch_norm.get_weights()[0]
                self.assertTrue(np.any(gamma != 0.0))

    def test_interface_and_branch_fusion_parity(self):
        import numpy as np

        from models.hand_landmarker.registry import build_model, reparameterize_for_deploy

        iterations = [1, 1, 1, 1, 1, 1, 1]
        training_model = build_model("v2", num_iterations=iterations)
        self.assertEqual((None, 1, 256, 256), tuple(training_model.input_shape))
        self.assertEqual(
            [(None, 1, 1, 42), (None, 1, 1, 1), (None, 1, 1, 1)],
            [tuple(shape) for shape in training_model.output_shape],
        )
        self.assertEqual(
            ["convld_21_2d", "activation_handflag", "activation_handedness"],
            list(training_model.output_names),
        )
        tensor = np.random.RandomState(7).uniform(0, 1, (1, 1, 256, 256)).astype("float32")
        initial = training_model(tensor, training=False)
        self.assertTrue(np.allclose(initial[0], 0.5))
        for scalar_head in initial[1:]:
            self.assertTrue(np.all(np.isfinite(scalar_head)))
            self.assertLess(float(np.max(np.abs(scalar_head - 0.5))), 0.1)

        # Make every folded BN and all three heads non-trivial so parity cannot
        # pass merely because the initial head kernels are zero.
        random = np.random.RandomState(11)
        for layer in training_model.layers:
            bn = getattr(layer, "train_bn", None)
            if bn is None:
                continue
            gamma, beta, mean, variance = bn.get_weights()
            bn.set_weights(
                [
                    random.uniform(0.5, 1.5, gamma.shape).astype("float32"),
                    random.normal(0.0, 0.1, beta.shape).astype("float32"),
                    random.normal(0.0, 0.1, mean.shape).astype("float32"),
                    random.uniform(0.5, 1.5, variance.shape).astype("float32"),
                ]
            )
        for name in ("convld_21_2d", "conv_handflag", "conv_handedness"):
            layer = training_model.get_layer(name)
            kernel, bias = layer.get_weights()
            layer.set_weights(
                [
                    random.normal(0.0, 0.02, kernel.shape).astype("float32"),
                    random.normal(0.0, 0.02, bias.shape).astype("float32"),
                ]
            )

        deploy_model = reparameterize_for_deploy(
            training_model, version="v2", num_iterations=iterations
        )
        self.assertLess(deploy_model.count_params(), training_model.count_params())
        self.assertLess(deploy_model.count_params() * 4, 15 * 1024 * 1024)
        expected = training_model(tensor, training=False)
        actual = deploy_model(tensor, training=False)
        for left, right in zip(expected, actual):
            self.assertTrue(np.allclose(left, right, atol=5e-5, rtol=5e-4))


if __name__ == "__main__":
    unittest.main()
