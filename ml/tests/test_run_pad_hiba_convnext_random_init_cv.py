from __future__ import annotations

import unittest
from unittest.mock import patch

from ml.training.run_pad_hiba_convnext_random_init_cv import (
    RIGHTS_CLEAN_BATCH_SIZE,
    RIGHTS_CLEAN_EPOCHS,
    RIGHTS_CLEAN_LEARNING_RATE,
    RIGHTS_CLEAN_SEED,
    RIGHTS_CLEAN_WEIGHT_DECAY,
    parse_args,
)


class PadHibaConvnextRandomInitializationTests(unittest.TestCase):
    def test_protocol_keeps_the_locked_reference_hyperparameters(self) -> None:
        self.assertEqual(RIGHTS_CLEAN_EPOCHS, 15)
        self.assertEqual(RIGHTS_CLEAN_BATCH_SIZE, 32)
        self.assertEqual(RIGHTS_CLEAN_LEARNING_RATE, 1e-4)
        self.assertEqual(RIGHTS_CLEAN_WEIGHT_DECAY, 1e-4)
        self.assertEqual(RIGHTS_CLEAN_SEED, 42)

    def test_cli_does_not_expose_protocol_tuning(self) -> None:
        with patch("sys.argv", ["run_pad_hiba_convnext_random_init_cv"]):
            args = parse_args()

        self.assertEqual(args.device, "cuda")
        self.assertFalse(args.resume)
        self.assertFalse(hasattr(args, "weights"))
        self.assertFalse(hasattr(args, "epochs"))
        self.assertFalse(hasattr(args, "learning_rate"))


if __name__ == "__main__":
    unittest.main()
