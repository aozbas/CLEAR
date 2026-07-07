import unittest

from ml.evaluation.metrics import confusion_matrix, per_class_metrics, summarize_metrics
from ml.evaluation.schema import HAM10000_LABELS


class EvaluationMetricsTests(unittest.TestCase):
    def test_confusion_matrix_uses_canonical_row_and_column_order(self) -> None:
        matrix = confusion_matrix(
            ["melanoma", "nevus", "vascular_lesion"],
            ["nevus", "nevus", "melanoma"],
        )

        self.assertEqual(len(matrix), len(HAM10000_LABELS))
        self.assertEqual(len(matrix[0]), len(HAM10000_LABELS))
        self.assertEqual(matrix[0][1], 1)
        self.assertEqual(matrix[1][1], 1)
        self.assertEqual(matrix[6][0], 1)

    def test_perfect_predictions_have_perfect_summary_metrics(self) -> None:
        truth = list(HAM10000_LABELS)
        summary = summarize_metrics(truth, truth)

        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["balanced_accuracy"], 1.0)
        self.assertEqual(summary["macro_precision"], 1.0)
        self.assertEqual(summary["macro_recall"], 1.0)
        self.assertEqual(summary["macro_f1"], 1.0)

    def test_zero_predicted_positives_get_zero_precision_and_recall(self) -> None:
        matrix = confusion_matrix(
            ["melanoma", "nevus", "nevus"],
            ["nevus", "nevus", "nevus"],
        )
        metrics = per_class_metrics(matrix)

        self.assertEqual(metrics["melanoma"]["support"], 1)
        self.assertEqual(metrics["melanoma"]["predicted"], 0)
        self.assertEqual(metrics["melanoma"]["precision"], 0.0)
        self.assertEqual(metrics["melanoma"]["recall"], 0.0)
        self.assertEqual(metrics["melanoma"]["f1"], 0.0)

    def test_zero_true_examples_get_zero_recall(self) -> None:
        matrix = confusion_matrix(
            ["melanoma", "nevus"],
            ["melanoma", "basal_cell_carcinoma"],
        )
        metrics = per_class_metrics(matrix)

        self.assertEqual(metrics["basal_cell_carcinoma"]["support"], 0)
        self.assertEqual(metrics["basal_cell_carcinoma"]["predicted"], 1)
        self.assertEqual(metrics["basal_cell_carcinoma"]["precision"], 0.0)
        self.assertEqual(metrics["basal_cell_carcinoma"]["recall"], 0.0)

    def test_macro_metrics_include_all_canonical_classes(self) -> None:
        summary = summarize_metrics(
            ["melanoma", "nevus", "nevus"],
            ["nevus", "nevus", "nevus"],
        )

        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertAlmostEqual(summary["balanced_accuracy"], 1 / len(HAM10000_LABELS))
        self.assertAlmostEqual(summary["macro_precision"], (2 / 3) / len(HAM10000_LABELS))
        self.assertAlmostEqual(summary["macro_recall"], 1 / len(HAM10000_LABELS))
        self.assertAlmostEqual(summary["macro_f1"], 0.8 / len(HAM10000_LABELS))

    def test_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            summarize_metrics(["nevus"], [])


if __name__ == "__main__":
    unittest.main()
