import sys
import unittest
from pathlib import Path

from ml.training.run_pad_ufes_cv import build_training_command


class RunPadUfesCrossValidationTests(unittest.TestCase):
    def test_build_command_locks_image_only_baseline_configuration(self) -> None:
        command = build_training_command(
            split_csv=Path("fold_0.csv"),
            checkpoint=Path("fold_0.pt"),
            run_dir=Path("fold_0"),
            epochs=15,
            batch_size=32,
            lr=1e-4,
            weight_decay=1e-4,
            augmentation_profile="regularized_v2",
            label_smoothing=0.1,
            lr_schedule="cosine",
            num_workers=0,
            seed=42,
            device="cuda",
        )

        self.assertEqual(command[:4], [sys.executable, "-u", "-m", "ml.training.train_pad_ufes"])
        self.assertEqual(command[command.index("--weights") + 1], "imagenet")
        self.assertEqual(command[command.index("--epochs") + 1], "15")
        self.assertEqual(
            command[command.index("--augmentation-profile") + 1],
            "regularized_v2",
        )
        self.assertEqual(command[command.index("--label-smoothing") + 1], "0.1")
        self.assertEqual(command[command.index("--lr-schedule") + 1], "cosine")
        self.assertEqual(command[command.index("--seed") + 1], "42")
        self.assertEqual(command[command.index("--device") + 1], "cuda")


if __name__ == "__main__":
    unittest.main()
