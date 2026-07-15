import tempfile
import unittest
from pathlib import Path

from scripts.build_export_preflight import build_preflight_bundle


class ExportPreflightGuardTests(unittest.TestCase):
    def test_requires_explicit_preflight_flag_before_importing_tensorflow(self):
        with self.assertRaisesRegex(ValueError, "preflight_untrained"):
            build_preflight_bundle({"export": {"overwrite": True}})

    def test_refuses_random_weights_outside_preflight_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "hand": {"model_path": str(Path(directory) / "model.weights.h5")},
                "export": {"preflight_untrained": True, "overwrite": True},
                "_meta": {"repo_root": directory},
            }
            with self.assertRaisesRegex(ValueError, "preflight/untrained"):
                build_preflight_bundle(config)


if __name__ == "__main__":
    unittest.main()
