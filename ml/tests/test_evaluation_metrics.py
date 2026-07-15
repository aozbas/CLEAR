import unittest

from ml.evaluation.metrics import confusion_matrix, per_class_metrics, summarize_metrics
from ml.evaluation.schema import HAM10000_LABELS, PAD_UFES_NATIVE_LABELS


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

    def test_covered_label_macro_metrics_exclude_uncovered_labels(self) -> None:
        summary = summarize_metrics(
            ["melanoma", "melanoma", "nevus", "nevus"],
            ["melanoma", "nevus", "nevus", "basal_cell_carcinoma"],
        )

        self.assertEqual(summary["covered_labels"], ["melanoma", "nevus"])
        self.assertAlmostEqual(summary["covered_label_macro_precision"], 0.75)
        self.assertAlmostEqual(summary["covered_label_macro_recall"], 0.5)
        self.assertAlmostEqual(summary["covered_label_balanced_accuracy"], 0.5)
        self.assertAlmostEqual(summary["covered_label_macro_f1"], 7 / 12)

    def test_prediction_distribution_counts_all_canonical_labels(self) -> None:
        summary = summarize_metrics(
            ["melanoma", "melanoma", "nevus", "nevus"],
            ["melanoma", "nevus", "nevus", "basal_cell_carcinoma"],
        )

        distribution = summary["prediction_distribution"]
        self.assertEqual(set(distribution), set(HAM10000_LABELS))
        self.assertEqual(distribution["melanoma"], {"count": 1, "fraction": 0.25})
        self.assertEqual(distribution["nevus"], {"count": 2, "fraction": 0.5})
        self.assertEqual(
            distribution["basal_cell_carcinoma"],
            {"count": 1, "fraction": 0.25},
        )
        self.assertEqual(distribution["vascular_lesion"], {"count": 0, "fraction": 0.0})

    def test_summary_accepts_pad_ufes_native_label_set(self) -> None:
        summary = summarize_metrics(
            ["squamous_cell_carcinoma", "seborrheic_keratosis", "melanoma"],
            ["squamous_cell_carcinoma", "melanoma", "melanoma"],
            labels=PAD_UFES_NATIVE_LABELS,
        )

        self.assertEqual(summary["labels"], list(PAD_UFES_NATIVE_LABELS))
        self.assertEqual(
            summary["covered_labels"],
            ["melanoma", "squamous_cell_carcinoma", "seborrheic_keratosis"],
        )
        self.assertEqual(
            set(summary["prediction_distribution"]),
            set(PAD_UFES_NATIVE_LABELS),
        )
        self.assertEqual(summary["per_class"]["squamous_cell_carcinoma"]["true_positive"], 1)

    def test_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            summarize_metrics(["nevus"], [])


if __name__ == "__main__":
    unittest.main()
