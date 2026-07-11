"""Formal local model-promotion gate for experimental evaluation reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from ml.evaluation.stress import PHONE_STRESS_VARIANTS

REQUIRED_DATASET_KEYS = ("ham10000_internal", "ph2_holdout", "derm7pt_holdout")
GATE_PROFILES = {
    "dermoscopy": REQUIRED_DATASET_KEYS,
    "phone_clinical": (
        "ham10000_internal",
        "ph2_holdout",
        "derm7pt_holdout",
        "pad_ufes_clinical",
    ),
}
REQUIRED_PHONE_STRESS_VARIANTS = tuple(PHONE_STRESS_VARIANTS)

DEFAULT_THRESHOLDS = {
    "ham10000_macro_f1": 0.28,
    "ham10000_balanced_accuracy": 0.32,
    "external_melanoma_recall": 0.50,
    "external_covered_label_macro_f1": 0.40,
    "max_latency_p95_ms": 250.0,
    "max_single_label_prediction_fraction": 0.90,
    "min_melanoma_prediction_fraction": 0.05,
    "max_phone_stress_relative_drop": 0.25,
}


def load_report_bundle(report_dirs: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    reports: dict[str, dict[str, object]] = {}
    for dataset_key, report_dir in report_dirs.items():
        report_path = Path(report_dir)
        metrics = json.loads((report_path / "metrics.json").read_text(encoding="utf-8"))
        metadata = json.loads((report_path / "metadata.json").read_text(encoding="utf-8"))
        reports[dataset_key] = {
            "metrics": metrics,
            "metadata": metadata,
            "report_dir": str(report_path),
        }
    return reports


def evaluate_promotion_gate(
    reports: Mapping[str, Mapping[str, object]],
    *,
    thresholds: Mapping[str, float] | None = None,
    profile: str = "dermoscopy",
) -> dict[str, object]:
    try:
        required_dataset_keys = GATE_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown promotion gate profile: {profile}") from exc

    active_thresholds = {**DEFAULT_THRESHOLDS, **(dict(thresholds or {}))}
    checks: list[dict[str, object]] = []

    for dataset_key in required_dataset_keys:
        report = reports.get(dataset_key)
        if report is None:
            checks.append(
                _check(
                    name=f"{dataset_key} report present",
                    status="fail",
                    detail="Required evaluation report is missing.",
                )
            )
            continue
        _check_report_identity(checks, dataset_key, report)
        _check_latency(checks, dataset_key, report, active_thresholds)
        _check_prediction_distribution(checks, dataset_key, report, active_thresholds)
        _check_contamination_notes(checks, dataset_key, report)

    ham_report = reports.get("ham10000_internal")
    if ham_report is not None:
        _check_metric_at_least(
            checks,
            name="ham10000_internal macro_f1",
            observed=_metric(ham_report, "macro_f1"),
            threshold=active_thresholds["ham10000_macro_f1"],
        )
        _check_metric_at_least(
            checks,
            name="ham10000_internal balanced_accuracy",
            observed=_metric(ham_report, "balanced_accuracy"),
            threshold=active_thresholds["ham10000_balanced_accuracy"],
        )
        _check_phone_stress(checks, ham_report, active_thresholds)

    for dataset_key in required_dataset_keys:
        if dataset_key == "ham10000_internal":
            continue
        report = reports.get(dataset_key)
        if report is None:
            continue
        _check_metric_at_least(
            checks,
            name=f"{dataset_key} melanoma recall",
            observed=_melanoma_recall(_metrics(report)),
            threshold=active_thresholds["external_melanoma_recall"],
        )
        _check_metric_at_least(
            checks,
            name=f"{dataset_key} covered_label_macro_f1",
            observed=_metric(report, "covered_label_macro_f1"),
            threshold=active_thresholds["external_covered_label_macro_f1"],
        )

    decision = "block" if any(check["status"] == "fail" for check in checks) else "promote"
    return {
        "decision": decision,
        "profile": profile,
        "thresholds": active_thresholds,
        "required_datasets": list(required_dataset_keys),
        "required_phone_stress_variants": list(REQUIRED_PHONE_STRESS_VARIANTS),
        "checks": checks,
    }


def write_promotion_gate(output_dir: Path, result: Mapping[str, object]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _summary_markdown(result),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="DATASET_KEY=REPORT_DIR",
        help="Evaluation report directory for a required dataset source.",
    )
    parser.add_argument("--profile", choices=sorted(GATE_PROFILES), default="dermoscopy")
    parser.add_argument("--out", type=Path, required=True)
    try:
        args = parser.parse_args(argv)
        report_dirs = _parse_report_dirs(args.report)
        result = evaluate_promotion_gate(load_report_bundle(report_dirs), profile=args.profile)
        write_promotion_gate(args.out, result)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Wrote model promotion gate to {args.out} (decision={result['decision']})")
    return 0


def _parse_report_dirs(report_args: list[str]) -> dict[str, Path]:
    report_dirs: dict[str, Path] = {}
    for report_arg in report_args:
        if "=" not in report_arg:
            raise ValueError("--report must use DATASET_KEY=REPORT_DIR.")
        dataset_key, report_dir = report_arg.split("=", 1)
        if not dataset_key:
            raise ValueError("--report dataset key cannot be empty.")
        if not report_dir:
            raise ValueError("--report directory cannot be empty.")
        report_dirs[dataset_key] = Path(report_dir)
    return report_dirs


def _check_report_identity(
    checks: list[dict[str, object]],
    expected_dataset_key: str,
    report: Mapping[str, object],
) -> None:
    dataset = _dataset_metadata(report)
    observed = dataset.get("key")
    status = "pass" if observed == expected_dataset_key else "fail"
    checks.append(
        _check(
            name=f"{expected_dataset_key} report identity",
            status=status,
            observed=observed,
            threshold=expected_dataset_key,
            detail=(
                "Report dataset key matches required source."
                if status == "pass"
                else "Report dataset key does not match the required source."
            ),
        )
    )


def _check_latency(
    checks: list[dict[str, object]],
    dataset_key: str,
    report: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> None:
    _check_metric_at_most(
        checks,
        name=f"{dataset_key} latency_p95_ms",
        observed=_metric(report, "latency_p95_ms"),
        threshold=thresholds["max_latency_p95_ms"],
    )


def _check_prediction_distribution(
    checks: list[dict[str, object]],
    dataset_key: str,
    report: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> None:
    metrics = _metrics(report)
    distribution = metrics.get("prediction_distribution")
    if not isinstance(distribution, Mapping):
        checks.append(
            _check(
                name=f"{dataset_key} prediction distribution present",
                status="fail",
                detail="Prediction distribution is missing.",
            )
        )
        return

    fractions = []
    for label_distribution in distribution.values():
        if isinstance(label_distribution, Mapping):
            fraction = _as_float(label_distribution.get("fraction"))
            if fraction is not None:
                fractions.append(fraction)
    max_fraction = max(fractions) if fractions else None
    _check_metric_at_most(
        checks,
        name=f"{dataset_key} dominant prediction share",
        observed=max_fraction,
        threshold=thresholds["max_single_label_prediction_fraction"],
    )

    if _melanoma_support(metrics) > 0:
        melanoma_distribution = distribution.get("melanoma")
        melanoma_fraction = None
        if isinstance(melanoma_distribution, Mapping):
            melanoma_fraction = _as_float(melanoma_distribution.get("fraction"))
        _check_metric_at_least(
            checks,
            name=f"{dataset_key} melanoma prediction share",
            observed=melanoma_fraction,
            threshold=thresholds["min_melanoma_prediction_fraction"],
        )


def _check_contamination_notes(
    checks: list[dict[str, object]],
    dataset_key: str,
    report: Mapping[str, object],
) -> None:
    dataset = _dataset_metadata(report)
    notes = dataset.get("contamination_notes", [])
    if not isinstance(notes, list) or not notes:
        checks.append(
            _check(
                name=f"{dataset_key} contamination notes",
                status="pass",
                observed=[],
                detail="No known contamination notes were recorded.",
            )
        )
        return

    split_type = str(dataset.get("split_type", ""))
    status = "warn" if split_type == "internal_test" else "fail"
    checks.append(
        _check(
            name=f"{dataset_key} contamination notes",
            status=status,
            observed=notes,
            detail=(
                "Internal-source contamination note retained for review."
                if status == "warn"
                else "External holdout contamination notes block promotion."
            ),
        )
    )


def _check_phone_stress(
    checks: list[dict[str, object]],
    ham_report: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> None:
    metrics = _metrics(ham_report)
    phone_stress = metrics.get("phone_stress")
    if not isinstance(phone_stress, Mapping):
        checks.append(
            _check(
                name="phone stress present",
                status="fail",
                detail="HAM10000 report is missing opt-in phone-stress metrics.",
            )
        )
        return

    checks.append(
        _check(
            name="phone stress present",
            status="pass",
            detail="HAM10000 report includes opt-in phone-stress metrics.",
        )
    )

    variants = phone_stress.get("variants")
    if not isinstance(variants, Mapping):
        checks.append(
            _check(
                name="phone stress variants present",
                status="fail",
                detail="Phone-stress variant metrics are missing.",
            )
        )
        return

    clean_melanoma_recall = _melanoma_recall(metrics)
    clean_covered_f1 = _as_float(metrics.get("covered_label_macro_f1"))
    max_drop = thresholds["max_phone_stress_relative_drop"]
    for variant_key in REQUIRED_PHONE_STRESS_VARIANTS:
        variant_metrics = variants.get(variant_key)
        if not isinstance(variant_metrics, Mapping):
            checks.append(
                _check(
                    name=f"phone stress {variant_key} present",
                    status="fail",
                    detail="Required phone-stress variant is missing.",
                )
            )
            continue

        if clean_melanoma_recall is not None:
            _check_metric_at_least(
                checks,
                name=f"phone stress {variant_key} melanoma recall",
                observed=_melanoma_recall(variant_metrics),
                threshold=clean_melanoma_recall * (1.0 - max_drop),
            )
        if clean_covered_f1 is not None:
            _check_metric_at_least(
                checks,
                name=f"phone stress {variant_key} covered_label_macro_f1",
                observed=_as_float(variant_metrics.get("covered_label_macro_f1")),
                threshold=clean_covered_f1 * (1.0 - max_drop),
            )
        _check_metric_at_most(
            checks,
            name=f"phone stress {variant_key} latency_p95_ms",
            observed=_as_float(variant_metrics.get("latency_p95_ms")),
            threshold=thresholds["max_latency_p95_ms"],
        )


def _check_metric_at_least(
    checks: list[dict[str, object]],
    *,
    name: str,
    observed: float | None,
    threshold: float,
) -> None:
    status = "pass" if observed is not None and observed >= threshold else "fail"
    checks.append(
        _check(
            name=name,
            status=status,
            observed=observed,
            threshold=threshold,
            detail=(
                "Observed value meets or exceeds the threshold."
                if status == "pass"
                else "Observed value is missing or below the threshold."
            ),
        )
    )


def _check_metric_at_most(
    checks: list[dict[str, object]],
    *,
    name: str,
    observed: float | None,
    threshold: float,
) -> None:
    status = "pass" if observed is not None and observed <= threshold else "fail"
    checks.append(
        _check(
            name=name,
            status=status,
            observed=observed,
            threshold=threshold,
            detail=(
                "Observed value is within the threshold."
                if status == "pass"
                else "Observed value is missing or above the threshold."
            ),
        )
    )


def _check(
    *,
    name: str,
    status: str,
    detail: str,
    observed: object | None = None,
    threshold: object | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "status": status,
        "observed": observed,
        "threshold": threshold,
        "detail": detail,
    }


def _metrics(report: Mapping[str, object]) -> Mapping[str, object]:
    metrics = report.get("metrics", {})
    return metrics if isinstance(metrics, Mapping) else {}


def _dataset_metadata(report: Mapping[str, object]) -> Mapping[str, object]:
    metadata = report.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return {}
    dataset = metadata.get("dataset", {})
    return dataset if isinstance(dataset, Mapping) else {}


def _metric(report: Mapping[str, object], name: str) -> float | None:
    return _as_float(_metrics(report).get(name))


def _melanoma_recall(metrics: Mapping[str, object]) -> float | None:
    per_class = metrics.get("per_class")
    if not isinstance(per_class, Mapping):
        return None
    melanoma = per_class.get("melanoma")
    if not isinstance(melanoma, Mapping):
        return None
    return _as_float(melanoma.get("recall"))


def _melanoma_support(metrics: Mapping[str, object]) -> int:
    per_class = metrics.get("per_class")
    if not isinstance(per_class, Mapping):
        return 0
    melanoma = per_class.get("melanoma")
    if not isinstance(melanoma, Mapping):
        return 0
    support = _as_float(melanoma.get("support"))
    return int(support or 0)


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _summary_markdown(result: Mapping[str, object]) -> str:
    checks = result.get("checks", [])
    rows = []
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            rows.append(
                "| "
                f"{check.get('status', '')} | "
                f"{check.get('name', '')} | "
                f"{_format_value(check.get('observed'))} | "
                f"{_format_value(check.get('threshold'))} | "
                f"{check.get('detail', '')} |"
            )

    return "\n".join(
        [
            "# Model Promotion Gate",
            "",
            "This gate records experimental classification evidence only. "
            "It is not a medical diagnosis.",
            "",
            f"Decision: {result.get('decision', 'block')}",
            "",
            "| Status | Check | Observed | Threshold | Detail |",
            "|---|---|---:|---:|---|",
            *rows,
            "",
        ]
    )


def _format_value(value: object) -> str:
    if isinstance(value, float):
        formatted = f"{value:.4f}".rstrip("0").rstrip(".")
        return formatted if formatted else "0"
    if value is None:
        return ""
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
