import json
import tempfile
import unittest
from pathlib import Path

from hand_landmarker.io_utils import write_jsonl
from scripts.prepare_finetune_round import (
    HARD_GOLD_LIMIT,
    _new_recorded_counts,
    _validate_gold_budget,
)


class FinetuneRoundTests(unittest.TestCase):
    def test_arbitrary_budget_within_hard_limit_is_valid(self):
        self.assertEqual(450, _validate_gold_budget(450))
        self.assertEqual(HARD_GOLD_LIMIT, _validate_gold_budget(HARD_GOLD_LIMIT))

    def test_budget_outside_hard_limit_is_rejected(self):
        for value in (0, -1, HARD_GOLD_LIMIT + 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "gold-budget"):
                    _validate_gold_budget(value)

    def test_multiple_new_recorded_tasks_share_one_round_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "GoldSource"
            for source_id, count in (("r01", 2), ("r02", 3)):
                task = root / "new_recorded_gold" / source_id / "task"
                manifest = task / "02_roi_crops/hand_roi_crops_manifest.jsonl"
                write_jsonl(manifest, [{"crop_id": f"{source_id}:{i}"} for i in range(count)])
                (task / "task_descriptor.json").write_text(
                    json.dumps(
                        {
                            "source_id": source_id,
                            "source_kind": "new_recorded_gold",
                            "artifacts": {
                                "manifest": {
                                    "path": "02_roi_crops/hand_roi_crops_manifest.jsonl"
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            total, report = _new_recorded_counts(
                {"gold_repository_root": str(root)}, ["r01", "r02"], 5
            )
            self.assertEqual(5, total)
            self.assertEqual(["r01", "r02"], [row["source_id"] for row in report["sources"]])
            with self.assertRaisesRegex(ValueError, "exceed"):
                _new_recorded_counts(
                    {"gold_repository_root": str(root)}, ["r01", "r02"], 4
                )


if __name__ == "__main__":
    unittest.main()
