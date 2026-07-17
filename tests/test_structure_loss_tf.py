from __future__ import annotations

import unittest

from hand_landmarker.training import _build_weighted_trainer


try:
    import numpy as np
    import tensorflow as tf
except ImportError:  # Local inspection environments do not require TensorFlow.
    np = None
    tf = None


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class StructureLossTensorFlowTests(unittest.TestCase):
    @staticmethod
    def _model():
        inputs = tf.keras.Input(shape=(4,))
        hidden = tf.keras.layers.Dense(16, activation="relu")(inputs)
        landmarks = tf.keras.layers.Dense(42, activation="sigmoid", name="landmarks")(hidden)
        hand_flag = tf.keras.layers.Dense(1, activation="sigmoid", name="hand_flag")(hidden)
        handedness = tf.keras.layers.Dense(1, activation="sigmoid", name="handedness")(hidden)
        return tf.keras.Model(inputs, [landmarks, hand_flag, handedness])

    @staticmethod
    def _losses(bone: float = 5.0, spread: float = 1.0):
        return {
            "landmarks": {"coefficient": 1.0, "delta": 0.02},
            "hand_flag": {"coefficient": 0.1},
            "handedness": {"coefficient": 0.1},
            "bone_vector": {"coefficient": bone, "delta": 0.02},
            "spread_ratio": {"coefficient": spread, "delta": 0.1},
        }

    @staticmethod
    def _batch():
        x = np.ones((3, 4), dtype="float32")
        points = np.linspace(0.1, 0.9, 42, dtype="float32")
        y = {
            "landmarks": np.stack([points, points * 0.8, points * 0.6]),
            "hand_flag": np.ones((3, 1), dtype="float32"),
            "handedness": np.asarray([[0.0], [1.0], [0.0]], dtype="float32"),
        }
        weights = {
            "landmarks": np.ones(3, dtype="float32"),
            "hand_flag": np.ones(3, dtype="float32"),
            "handedness": np.ones(3, dtype="float32"),
            # Only the first record represents human-reviewed Gold supervision.
            "structure": np.asarray([1.0, 0.0, 0.0], dtype="float32"),
        }
        return x, y, weights

    def test_fit_accepts_structure_mask_and_produces_finite_losses(self) -> None:
        trainer = _build_weighted_trainer(tf, self._model(), self._losses())
        trainer.compile(optimizer=tf.keras.optimizers.Adam(1.0e-3), run_eagerly=False)
        x, y, weights = self._batch()
        history = trainer.fit(x, y, sample_weight=weights, batch_size=3, epochs=1, verbose=0)
        self.assertTrue(np.isfinite(history.history["total_loss"][-1]))
        self.assertGreaterEqual(history.history["bone_vector_loss"][-1], 0.0)
        self.assertGreaterEqual(history.history["spread_ratio_loss"][-1], 0.0)

    def test_enabled_structure_losses_fail_closed_without_gold_mask(self) -> None:
        trainer = _build_weighted_trainer(tf, self._model(), self._losses())
        trainer.compile(optimizer=tf.keras.optimizers.Adam(1.0e-3), run_eagerly=True)
        x, y, weights = self._batch()
        weights.pop("structure")
        with self.assertRaisesRegex(ValueError, "explicit per-record structure mask"):
            trainer.train_on_batch(x, y, sample_weight=weights)


if __name__ == "__main__":
    unittest.main()
