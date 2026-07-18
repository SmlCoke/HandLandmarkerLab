import unittest

from scripts.prepare_finetune_round import HARD_GOLD_LIMIT, _validate_gold_budget


class FinetuneRoundTests(unittest.TestCase):
    def test_arbitrary_budget_within_hard_limit_is_valid(self):
        self.assertEqual(450, _validate_gold_budget(450))
        self.assertEqual(HARD_GOLD_LIMIT, _validate_gold_budget(HARD_GOLD_LIMIT))

    def test_budget_outside_hard_limit_is_rejected(self):
        for value in (0, -1, HARD_GOLD_LIMIT + 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "gold-budget"):
                    _validate_gold_budget(value)


if __name__ == "__main__":
    unittest.main()
