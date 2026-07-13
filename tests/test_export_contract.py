import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from hand_landmarker.export import _a1_attribute_audit, _guard_export_outputs


class _Helper:
    @staticmethod
    def get_attribute_value(attribute):
        return attribute.value


def _attribute(name, value):
    return SimpleNamespace(name=name, value=value)


def _graph(nodes, weights):
    initializers = [SimpleNamespace(name=name, dims=dims) for name, dims in weights.items()]
    return SimpleNamespace(graph=SimpleNamespace(node=nodes, initializer=initializers))


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
            SimpleNamespace(
                name="activation",
                op_type="LeakyRelu",
                input=["x"],
                attribute=[_attribute("alpha", 0.1)],
            ),
        ]
        report = _a1_attribute_audit(
            _graph(nodes, {"w": [16, 8, 3, 3]}), SimpleNamespace(helper=_Helper)
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
            SimpleNamespace(
                name="activation",
                op_type="LeakyRelu",
                input=["x"],
                attribute=[_attribute("alpha", 0.2)],
            ),
        ]
        report = _a1_attribute_audit(
            _graph(nodes, {"w": [16, 300, 3, 3]}), SimpleNamespace(helper=_Helper)
        )
        rules = [item["rule"] for item in report["violations"]]
        self.assertTrue(any("stride" in value for value in rules))
        self.assertTrue(any("2048" in value for value in rules))
        self.assertTrue(any("MaxPool" in value for value in rules))
        self.assertTrue(any("alpha" in value for value in rules))

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


if __name__ == "__main__":
    unittest.main()
