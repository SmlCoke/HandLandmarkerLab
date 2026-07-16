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
        return_value={
            "schema_version": "pretrain_curation_v1",
            "negative_review_transaction": {"status": "committed"},
        },
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
        return_value={
            "schema_version": "pretrain_curation_v1",
            "negative_review_transaction": {"status": "committed"},
        },
    )
    def test_unreviewed_negative_fails_closed(self, _verify):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "multitask.jsonl"
            write_jsonl(labels, [_positive(), _negative(confirmed=False)])
            report = check_multitask_data(self._config(labels))
            self.assertEqual("fail", report["status"])
            self.assertFalse(report["checks"]["no_unreviewed_negative"])

    @mock.patch(
        "scripts.check_multitask_data.verify_curation_manifest",
        return_value={"schema_version": "pretrain_curation_v1"},
    )
    def test_confirmed_negative_requires_transaction_authentication(self, _verify):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "multitask.jsonl"
            write_jsonl(labels, [_positive(), _negative()])
            report = check_multitask_data(self._config(labels))
            self.assertEqual("fail", report["status"])
            self.assertFalse(
                report["checks"]["authenticated_negative_review_transaction"]
            )

    @mock.patch(
        "scripts.check_multitask_data.verify_curation_manifest",
        return_value={
            "schema_version": "pretrain_curation_v1",
            "negative_review_transaction": {"status": "committed"},
        },
    )
    def test_gate_reports_exact_auto_epoch_resolution(self, _verify):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "multitask.jsonl"
            rows = []
            for crop_id, sample_type, negative in (
                ("pos-runtime", "POS_RUNTIME", False),
                ("pos-low", "POS_LOW_PALM", False),
                ("neg-runtime", "NEG_RUNTIME_CANDIDATE", True),
                ("neg-low", "NEG_LOW_PALM_CANDIDATE", True),
            ):
                row = _negative() if negative else _positive(crop_id)
                row["crop_id"] = crop_id
                row["sample_type"] = sample_type
                row["supervision_tier"] = "pseudo"
                row["sampling_bucket"] = "pseudo:{}".format(sample_type)
                row["sampling_weight"] = 1.0
                rows.append(row)
            write_jsonl(labels, rows)
            config = self._config(labels)
            config["training"] = {"batch_size": 4}
            config["sampling"] = {
                "epoch_size": "auto",
                "epoch_size_upper_bound": 8,
                "max_average_cell_draws_per_unique_record": 2.0,
                "max_expected_row_draws_per_epoch": 2.0,
                "sample_type_fractions": {
                    "POS_RUNTIME": 0.25,
                    "POS_LOW_PALM": 0.25,
                    "NEG_RUNTIME_CANDIDATE": 0.25,
                    "NEG_LOW_PALM_CANDIDATE": 0.25,
                },
            }
            report = check_multitask_data(config)
            self.assertEqual("pass", report["status"])
            self.assertEqual(8, report["resolved_epoch_size"])
            self.assertEqual(2.0, report["max_expected_row_draws"])
            self.assertTrue(
                all(
                    value == 2.0
                    for value in report["expected_draws_per_unique_record"].values()
                )
            )


if __name__ == "__main__":
    unittest.main()
