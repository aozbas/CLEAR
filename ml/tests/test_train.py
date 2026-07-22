import unittest
from pathlib import Path

import torch

from ml.training.train import class_weights, metrics_from_confusion, resolve_project_path


class TrainingUtilityTests(unittest.TestCase):
    def test_resolve_project_path_keeps_absolute_paths(self) -> None:
        path = Path.cwd().resolve() / "example.csv"

        self.assertEqual(resolve_project_path(path), path)

    def test_class_weights_balance_observed_counts(self) -> None:
        weights = class_weights([0, 0, 1], 2, torch.device("cpu"))

        torch.testing.assert_close(weights, torch.tensor([0.75, 1.5]))

    def test_class_weights_reject_missing_classes(self) -> None:
        with self.assertRaises(ValueError):
            class_weights([0, 0], 2, torch.device("cpu"))

    def test_metrics_from_confusion_reports_per_class_values(self) -> None:
        metrics = metrics_from_confusion(
            loss=0.5,
            accuracy=0.75,
            confusion=torch.tensor([[2, 1], [0, 1]]),
            label_names=["class_a", "class_b"],
        )

        self.assertEqual(metrics["loss"], 0.5)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["per_class"]["class_a"]["support"], 3)
        self.assertEqual(metrics["per_class"]["class_b"]["support"], 1)


if __name__ == "__main__":
    unittest.main()
