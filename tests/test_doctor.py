from types import SimpleNamespace
import unittest
from unittest import mock

from hand_landmarker import doctor


class _FakeTensorFlow:
    __version__ = "2.9.0"

    def __init__(self, built_with_cuda, gpus, build_info):
        self.test = SimpleNamespace(is_built_with_cuda=lambda: built_with_cuda)
        self.config = SimpleNamespace(list_physical_devices=lambda kind: list(gpus))
        self.sysconfig = SimpleNamespace(get_build_info=lambda: dict(build_info))


class DoctorGpuContractTests(unittest.TestCase):
    def _report(self, tensorflow, environment):
        with mock.patch.object(doctor, "PACKAGES", {"tensorflow": "tensorflow"}), mock.patch.object(
            doctor.importlib, "import_module", return_value=tensorflow
        ), mock.patch.object(
            doctor.metadata, "version", return_value=tensorflow.__version__
        ), mock.patch.object(
            doctor.sys, "version_info", (3, 8, 18)
        ), mock.patch.object(
            doctor.platform, "python_version", return_value="3.8.18"
        ):
            return doctor.environment_report({"environment": environment})

    def test_required_gpu_fails_for_cpu_tensorflow_and_no_device(self):
        report = self._report(
            _FakeTensorFlow(False, [], {"cuda_version": "11.2", "cudnn_version": "8"}),
            {
                "tensorflow": "==2.9.0",
                "cuda": "11.2",
                "cudnn_major": 8,
                "require_gpu": True,
            },
        )

        self.assertFalse(report["ok"])
        failures = "\n".join(report["failures"])
        self.assertIn("not built with CUDA support", failures)
        self.assertIn("found no physical GPU", failures)

    def test_build_match_does_not_claim_runtime_version_is_known(self):
        gpu = SimpleNamespace(name="/physical_device:GPU:0")
        report = self._report(
            _FakeTensorFlow(
                True,
                [gpu],
                {"cuda_version": "11.2", "cudnn_version": "8.1"},
            ),
            {
                "tensorflow": "==2.9.0",
                "cuda": "11.2",
                "cudnn_major": 8,
                "require_gpu": True,
            },
        )

        self.assertTrue(report["ok"])
        self.assertEqual("match", report["tensorflow"]["compatibility"]["cuda"]["build_status"])
        self.assertEqual("unknown", report["tensorflow"]["compatibility"]["cuda"]["runtime_version"])
        self.assertEqual("unknown", report["tensorflow"]["compatibility"]["cudnn"]["runtime_status"])
        warnings = "\n".join(report["warnings"])
        self.assertIn("CUDA runtime version cannot be determined reliably", warnings)
        self.assertIn("cuDNN runtime version cannot be determined reliably", warnings)

    def test_missing_build_versions_are_warnings_and_unknown(self):
        gpu = SimpleNamespace(name="/physical_device:GPU:0")
        report = self._report(
            _FakeTensorFlow(True, [gpu], {}),
            {
                "tensorflow": "==2.9.0",
                "cuda": "11.2",
                "cudnn_major": 8,
                "require_gpu": True,
            },
        )

        self.assertTrue(report["ok"])
        cuda = report["tensorflow"]["compatibility"]["cuda"]
        cudnn = report["tensorflow"]["compatibility"]["cudnn"]
        self.assertEqual(("unknown", "unknown"), (cuda["build_version"], cuda["build_status"]))
        self.assertEqual(("unknown", "unknown"), (cudnn["build_version"], cudnn["build_status"]))
        warnings = "\n".join(report["warnings"])
        self.assertIn("CUDA build version is unknown", warnings)
        self.assertIn("cuDNN build version is unknown", warnings)

    def test_mismatched_tensorflow_build_target_fails(self):
        gpu = SimpleNamespace(name="/physical_device:GPU:0")
        report = self._report(
            _FakeTensorFlow(
                True,
                [gpu],
                {"cuda_version": "11.8", "cudnn_version": "8"},
            ),
            {
                "tensorflow": "==2.9.0",
                "cuda": "11.2",
                "cudnn_major": 8,
                "require_gpu": True,
            },
        )

        self.assertFalse(report["ok"])
        self.assertIn(
            "TensorFlow CUDA build version mismatch: expected 11.2, found 11.8",
            report["failures"],
        )


if __name__ == "__main__":
    unittest.main()
