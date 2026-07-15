import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hand_landmarker.export import (
    _a1_attribute_audit,
    _guard_export_outputs,
    _quantization_readiness_audit,
    _validated_model_size,
)


class _Helper:
    @staticmethod
    def get_attribute_value(attribute):
        return attribute.value


def _attribute(name, value):
    return SimpleNamespace(name=name, value=value)


def _graph(nodes, weights):
    initializers = [
        SimpleNamespace(name=name, dims=value.shape, array=value)
        for name, value in weights.items()
    ]
    return SimpleNamespace(graph=SimpleNamespace(node=nodes, initializer=initializers))


def _weights(**shapes):
    return {
        name: np.ones(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }


ONNX_STUB = SimpleNamespace(
    helper=_Helper,
    numpy_helper=SimpleNamespace(to_array=lambda item: item.array),
)


class A1ExportAuditTests(unittest.TestCase):
    def test_valid_operator_attributes_pass(self):
        nodes = [
            SimpleNamespace(
                name="conv",
                op_type="Conv",
                input=["x", "w"],
                attribute=[
                    _attribute("kernel_shape", [3, 3]),
                    _attribute("strides", [2, 2]),
                    _attribute("pads", [1, 1, 1, 1]),
                ],
            ),
            SimpleNamespace(
                name="pool",
                op_type="MaxPool",
                input=["x"],
                attribute=[_attribute("kernel_shape", [2, 2])],
            ),
            SimpleNamespace(name="activation", op_type="Relu", input=["x"], attribute=[]),
        ]
        report = _a1_attribute_audit(
            _graph(nodes, _weights(w=(16, 8, 3, 3))), SimpleNamespace(helper=_Helper)
        )
        self.assertEqual(report["violations"], [])

    def test_invalid_a1_attributes_are_all_reported(self):
        nodes = [
            SimpleNamespace(
                name="conv",
                op_type="Conv",
                input=["x", "w"],
                attribute=[_attribute("strides", [17, 1])],
            ),
            SimpleNamespace(
                name="pool",
                op_type="MaxPool",
                input=["x"],
                attribute=[_attribute("kernel_shape", [9, 2])],
            ),
        ]
        report = _a1_attribute_audit(
            _graph(nodes, _weights(w=(16, 300, 3, 3))), SimpleNamespace(helper=_Helper)
        )
        rules = [item["rule"] for item in report["violations"]]
        self.assertTrue(any("stride" in value for value in rules))
        self.assertTrue(any("2048" in value for value in rules))
        self.assertTrue(any("MaxPool" in value for value in rules))
        self.assertEqual({"Conv", "MaxPool"}, set(report["checked_nodes"]))

    def test_quantization_readiness_rejects_zero_weights_and_excessive_group(self):
        node = SimpleNamespace(
            name="depthwise",
            op_type="Conv",
            input=["x", "w"],
            attribute=[_attribute("group", 192)],
        )
        graph = _graph(
            [node], {"w": np.zeros((192, 1, 3, 3), dtype=np.float32)}
        )
        report = _quantization_readiness_audit(graph, ONNX_STUB)
        rules = [item["rule"] for item in report["violations"]]
        self.assertTrue(any("entirely zero" in rule for rule in rules))
        self.assertTrue(any("exceeds" in rule for rule in rules))
        self.assertEqual(192, report["observed_maximum_group"])

    def test_quantization_readiness_accepts_nonzero_group_128(self):
        node = SimpleNamespace(
            name="depthwise",
            op_type="Conv",
            input=["x", "w"],
            attribute=[_attribute("group", 128)],
        )
        graph = _graph([node], _weights(w=(128, 1, 3, 3)))
        report = _quantization_readiness_audit(graph, ONNX_STUB)
        self.assertEqual([], report["violations"])
        self.assertEqual(128, report["observed_maximum_group"])

    def test_contract_and_onnx_share_the_same_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "hand.onnx"
            contract_path = root / "hand.contract.json"
            contract_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Export output already exists"):
                _guard_export_outputs(model_path, contract_path, overwrite=False)
            _guard_export_outputs(model_path, contract_path, overwrite=True)
            with self.assertRaisesRegex(ValueError, "must differ"):
                _guard_export_outputs(model_path, model_path, overwrite=True)

    def test_actual_onnx_size_is_hard_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            path.write_bytes(b"x" * 2048)
            size_bytes, size_mb = _validated_model_size(path, 1.0)
            self.assertEqual(2048, size_bytes)
            self.assertAlmostEqual(2048 / float(1024 * 1024), size_mb)
            with self.assertRaisesRegex(ValueError, "exceeding"):
                _validated_model_size(path, 0.001)
            with self.assertRaisesRegex(ValueError, "positive finite"):
                _validated_model_size(path, 0.0)


if __name__ == "__main__":
    unittest.main()
