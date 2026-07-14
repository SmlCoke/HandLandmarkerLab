import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hand_landmarker.io_utils import write_jsonl
from scripts.check_multitask_data import check_multitask_data


def _positive(crop_id="pos"):
    return {
        "crop_id": crop_id,
        "sample_type": "POS_RUNTIME",
        "hand_presence": {"present": True},
        "pretrain_curation": {"action": "INCLUDE_LANDMARKS"},
    }


def _negative(confirmed=True):
    return {
        "crop_id": "neg",
        "sample_type": "NEG_RUNTIME_CANDIDATE",
        "hand_presence": {"present": False},
        "pretrain_curation": {
            "action": "INCLUDE_CONFIRMED_NEGATIVE" if confirmed else "HOLD_NEGATIVE_CANDIDATE",
            "negative_evidence": "human_confirmed" if confirmed else None,
            "review": {
                "decision": "CONFIRMED_NEGATIVE" if confirmed else "HOLD",
                "reviewer": "tester",
                "reviewed_at": "2026-07-14T12:00:00+08:00",
                "review_method": "retained_after_visual_deletion_review",
                "review_image_sha256": "a" * 64,
            },
        },
    }


class MultitaskGateTests(unittest.TestCase):
    def _config(self, labels):
        return {
            "task": "train",
            "stage": "pretrain",
            "data": {"labels": str(labels), "curation_manifest": "unused"},
            "multitask_gate": {
                "minimum_confirmed_negatives": 1,
                "minimum_confirmed_by_sample_type": {"NEG_RUNTIME_CANDIDATE": 1},
                "require_review_method": "retained_after_visual_deletion_review",
                "require_review_fields": [
                    "reviewer",
                    "reviewed_at",
                    "review_method",
                    "review_image_sha256",
                ],
            },
        }

    @mock.patch(
        "scripts.check_multitask_data.verify_curation_manifest",
        return_value={"schema_version": "pretrain_curation_v1"},
    )
    def test_human_confirmed_negative_passes(self, _verify):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "multitask.jsonl"
            write_jsonl(labels, [_positive(), _negative()])
            report = check_multitask_data(self._config(labels))
            self.assertEqual("pass", report["status"])
            self.assertEqual(1, report["confirmed_negative_count"])

    @mock.patch(
        "scripts.check_multitask_data.verify_curation_manifest",
        return_value={"schema_version": "pretrain_curation_v1"},
    )
    def test_unreviewed_negative_fails_closed(self, _verify):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "multitask.jsonl"
            write_jsonl(labels, [_positive(), _negative(confirmed=False)])
            report = check_multitask_data(self._config(labels))
            self.assertEqual("fail", report["status"])
            self.assertFalse(report["checks"]["no_unreviewed_negative"])


if __name__ == "__main__":
    unittest.main()
