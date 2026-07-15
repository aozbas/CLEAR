import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ml.evaluation.promotion_gate import (
    REQUIRED_DATASET_KEYS,
    REQUIRED_PHONE_STRESS_VARIANTS,
    evaluate_promotion_gate,
    load_report_bundle,
    main,
    write_promotion_gate,
)
from ml.evaluation.schema import HAM10000_LABELS


class PromotionGateTests(unittest.TestCase):
    def test_gate_promotes_when_required_reports_and_stress_metrics_pass(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            report_dirs = {
                "ham10000_internal": self._write_report_dir(
                    root / "ham",
                    "ham10000_internal",
                    self._metrics(phone_stress=True),
                ),
                "ph2_holdout": self._write_report_dir(
                    root / "ph2",
                    "ph2_holdout",
                    self._metrics(melanoma_recall=0.62),
                ),
                "derm7pt_holdout": self._write_report_dir(
                    root / "derm7pt",
                    "derm7pt_holdout",
                    self._metrics(melanoma_recall=0.58),
                ),
            }

            bundle = load_report_bundle(report_dirs)
            result = evaluate_promotion_gate(bundle)
            output_dir = root / "gate"
            write_promotion_gate(output_dir, result)

            gate_json = json.loads((output_dir / "gate.json").read_text(encoding="utf-8"))
            summary = (output_dir / "summary.md").read_text(encoding="utf-8")

        self.assertEqual(set(bundle), set(REQUIRED_DATASET_KEYS))
        self.assertEqual(result["decision"], "promote")
        self.assertTrue(all(check["status"] != "fail" for check in result["checks"]))
        self.assertEqual(gate_json["decision"], "promote")
        self.assertIn("experimental classification", summary)

    def test_gate_blocks_high_accuracy_candidate_with_zero_external_melanoma_recall(self) -> None:
        reports = {
            "ham10000_internal": {
                "metrics": self._metrics(phone_stress=True),
                "metadata": self._metadata("ham10000_internal"),
            },
            "ph2_holdout": {
                "metrics": self._metrics(
                    accuracy=0.88,
                    covered_label_macro_f1=0.48,
                    melanoma_recall=0.0,
                    melanoma_prediction_fraction=0.0,
                ),
                "metadata": self._metadata("ph2_holdout"),
            },
            "derm7pt_holdout": {
                "metrics": self._metrics(melanoma_recall=0.58),
                "metadata": self._metadata("derm7pt_holdout"),
            },
        }

        result = evaluate_promotion_gate(reports)

        failed_names = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertEqual(result["decision"], "block")
        self.assertIn("ph2_holdout melanoma recall", failed_names)
        self.assertIn("ph2_holdout melanoma prediction share", failed_names)

    def test_gate_blocks_missing_phone_stress_results(self) -> None:
        reports = {
            "ham10000_internal": {
                "metrics": self._metrics(phone_stress=False),
                "metadata": self._metadata("ham10000_internal"),
            },
            "ph2_holdout": {
                "metrics": self._metrics(melanoma_recall=0.62),
                "metadata": self._metadata("ph2_holdout"),
            },
            "derm7pt_holdout": {
                "metrics": self._metrics(melanoma_recall=0.58),
                "metadata": self._metadata("derm7pt_holdout"),
            },
        }

        result = evaluate_promotion_gate(reports)

        failed_names = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertEqual(result["decision"], "block")
        self.assertIn("phone stress present", failed_names)

    def test_gate_fails_external_contamination_notes(self) -> None:
        reports = {
            "ham10000_internal": {
                "metrics": self._metrics(phone_stress=True),
                "metadata": self._metadata("ham10000_internal"),
            },
            "ph2_holdout": {
                "metrics": self._metrics(melanoma_recall=0.62),
                "metadata": self._metadata(
                    "ph2_holdout",
                    contamination_notes=["Known external overlap."],
                ),
            },
            "derm7pt_holdout": {
                "metrics": self._metrics(melanoma_recall=0.58),
                "metadata": self._metadata("derm7pt_holdout"),
            },
        }

        result = evaluate_promotion_gate(reports)

        self.assertEqual(result["decision"], "block")
        self.assertIn(
            "ph2_holdout contamination notes",
            [check["name"] for check in result["checks"] if check["status"] == "fail"],
        )

    def test_phone_clinical_profile_requires_pad_ufes_evidence(self) -> None:
        reports = {
            "ham10000_internal": {
                "metrics": self._metrics(phone_stress=True),
                "metadata": self._metadata("ham10000_internal"),
            },
            "ph2_holdout": {
                "metrics": self._metrics(melanoma_recall=0.62),
                "metadata": self._metadata("ph2_holdout"),
            },
            "derm7pt_holdout": {
                "metrics": self._metrics(melanoma_recall=0.58),
                "metadata": self._metadata("derm7pt_holdout"),
            },
        }

        result = evaluate_promotion_gate(reports, profile="phone_clinical")

        failed_names = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertEqual(result["decision"], "block")
        self.assertIn("pad_ufes_clinical report present", failed_names)
        self.assertIn("pad_ufes_clinical", result["required_datasets"])

    def test_phone_clinical_profile_checks_pad_ufes_melanoma_recall(self) -> None:
        reports = {
            "ham10000_internal": {
                "metrics": self._metrics(phone_stress=True),
                "metadata": self._metadata("ham10000_internal"),
            },
            "ph2_holdout": {
                "metrics": self._metrics(melanoma_recall=0.62),
                "metadata": self._metadata("ph2_holdout"),
            },
            "derm7pt_holdout": {
                "metrics": self._metrics(melanoma_recall=0.58),
                "metadata": self._metadata("derm7pt_holdout"),
            },
            "pad_ufes_clinical": {
                "metrics": self._metrics(melanoma_recall=0.2, covered_label_macro_f1=0.35),
                "metadata": self._metadata("pad_ufes_clinical"),
            },
        }

        result = evaluate_promotion_gate(reports, profile="phone_clinical")

        failed_names = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertEqual(result["decision"], "block")
        self.assertIn("pad_ufes_clinical melanoma recall", failed_names)
        self.assertIn("pad_ufes_clinical covered_label_macro_f1", failed_names)

    def test_main_writes_gate_outputs_from_report_arguments(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            report_dirs = {
                "ham10000_internal": self._write_report_dir(
                    root / "ham",
                    "ham10000_internal",
                    self._metrics(phone_stress=True),
                ),
                "ph2_holdout": self._write_report_dir(
                    root / "ph2",
                    "ph2_holdout",
                    self._metrics(melanoma_recall=0.62),
                ),
                "derm7pt_holdout": self._write_report_dir(
                    root / "derm7pt",
                    "derm7pt_holdout",
                    self._metrics(melanoma_recall=0.58),
                ),
            }
            output_dir = root / "gate"

            exit_code = main(
                [
                    "--report",
                    f"ham10000_internal={report_dirs['ham10000_internal']}",
                    "--report",
                    f"ph2_holdout={report_dirs['ph2_holdout']}",
                    "--report",
                    f"derm7pt_holdout={report_dirs['derm7pt_holdout']}",
                    "--out",
                    str(output_dir),
                ]
            )

            gate_json = json.loads((output_dir / "gate.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(gate_json["decision"], "promote")

    def test_main_accepts_phone_clinical_profile(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            report_dirs = {
                "ham10000_internal": self._write_report_dir(
                    root / "ham",
                    "ham10000_internal",
                    self._metrics(phone_stress=True),
                ),
                "ph2_holdout": self._write_report_dir(
                    root / "ph2",
                    "ph2_holdout",
                    self._metrics(melanoma_recall=0.62),
                ),
                "derm7pt_holdout": self._write_report_dir(
                    root / "derm7pt",
                    "derm7pt_holdout",
                    self._metrics(melanoma_recall=0.58),
                ),
                "pad_ufes_clinical": self._write_report_dir(
                    root / "pad",
                    "pad_ufes_clinical",
                    self._metrics(melanoma_recall=0.62),
                ),
            }
            output_dir = root / "gate"

            exit_code = main(
                [
                    "--profile",
                    "phone_clinical",
                    "--report",
                    f"ham10000_internal={report_dirs['ham10000_internal']}",
                    "--report",
                    f"ph2_holdout={report_dirs['ph2_holdout']}",
                    "--report",
                    f"derm7pt_holdout={report_dirs['derm7pt_holdout']}",
                    "--report",
                    f"pad_ufes_clinical={report_dirs['pad_ufes_clinical']}",
                    "--out",
                    str(output_dir),
                ]
            )

            gate_json = json.loads((output_dir / "gate.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(gate_json["decision"], "promote")
        self.assertIn("pad_ufes_clinical", gate_json["required_datasets"])

    def _write_report_dir(
        self,
        path: Path,
        dataset_key: str,
        metrics: dict[str, object],
    ) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        (path / "metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
        (path / "metadata.json").write_text(
            json.dumps(self._metadata(dataset_key)),
            encoding="utf-8",
        )
        return path

    def _metadata(
        self,
        dataset_key: str,
        *,
        contamination_notes: list[str] | None = None,
    ) -> dict[str, object]:
        return {
            "model": {"name": "candidate"},
            "dataset": {
                "key": dataset_key,
                "name": dataset_key,
                "split_type": (
                    "internal_test" if dataset_key == "ham10000_internal" else "external_holdout"
                ),
                "contamination_notes": contamination_notes or [],
            },
        }

    def _metrics(
        self,
        *,
        accuracy: float = 0.72,
        macro_f1: float = 0.52,
        balanced_accuracy: float = 0.58,
        covered_label_macro_f1: float = 0.56,
        melanoma_recall: float = 0.56,
        melanoma_prediction_fraction: float = 0.18,
        dominant_fraction: float = 0.55,
        latency_p95_ms: float = 180.0,
        phone_stress: bool = False,
    ) -> dict[str, object]:
        distribution = {label: {"count": 0, "fraction": 0.0} for label in HAM10000_LABELS}
        distribution["melanoma"] = {
            "count": int(melanoma_prediction_fraction * 100),
            "fraction": melanoma_prediction_fraction,
        }
        distribution["nevus"] = {
            "count": int(dominant_fraction * 100),
            "fraction": dominant_fraction,
        }
        per_class = {
            label: {"support": 0, "predicted": 0, "recall": 0.0} for label in HAM10000_LABELS
        }
        per_class["melanoma"] = {
            "support": 20,
            "predicted": distribution["melanoma"]["count"],
            "recall": melanoma_recall,
        }
        metrics: dict[str, object] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_accuracy,
            "covered_label_macro_f1": covered_label_macro_f1,
            "covered_label_balanced_accuracy": 0.6,
            "latency_p95_ms": latency_p95_ms,
            "prediction_distribution": distribution,
            "per_class": per_class,
        }
        if phone_stress:
            metrics["phone_stress"] = self._phone_stress_metrics()
        return metrics

    def _phone_stress_metrics(self) -> dict[str, object]:
        variant_metrics = {}
        for variant in REQUIRED_PHONE_STRESS_VARIANTS:
            variant_metrics[variant] = {
                "sample_count": 20,
                "accuracy": 0.62,
                "covered_label_macro_f1": 0.48,
                "latency_p95_ms": 190.0,
                "per_class": {
                    "melanoma": {
                        "support": 20,
                        "recall": 0.48,
                    },
                },
            }
        return {
            "aggregate": {
                "sample_count": 20 * len(REQUIRED_PHONE_STRESS_VARIANTS),
                "accuracy": 0.62,
                "covered_label_macro_f1": 0.48,
                "latency_p95_ms": 190.0,
                "per_class": {"melanoma": {"support": 140, "recall": 0.48}},
            },
            "variants": variant_metrics,
        }


if __name__ == "__main__":
    unittest.main()
