"""Evaluate a post-hoc supported-input abstention gate for the fixed demo model.

This is development-only evidence. The fixed classifier saw every PAD/HIBA positive image during
its final fit, so positive retention is not independent model evidence. The report is aggregate and
must not contain paths, identifiers, per-image scores, or per-image decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from backend.app.services.image_validation import PoorImageQualityError, validate_image_quality
from ml.evaluation.prepare_open_images_ood import (
    CATEGORY_PLAN,
    EXCLUDED_HUMAN_CLASSES,
    EXPECTED_COHORT_FINGERPRINT,
    EXPECTED_LICENSE,
    EXPECTED_SOURCE_SHA256,
    IMAGES_PER_CATEGORY,
    MIN_TARGET_BOX_AREA,
    cohort_fingerprint,
    sha256_file,
)
from ml.evaluation.prepare_open_images_ood import (
    PROTOCOL as NEGATIVE_COHORT_PROTOCOL,
)
from ml.evaluation.prepare_open_images_ood import (
    SEED as NEGATIVE_COHORT_SEED,
)
from ml.inference.predict import DEFAULT_MODEL_PATH, load_model
from ml.preprocessing import get_transforms
from ml.training.run_pad_hiba_convnext_cv import (
    DEFAULT_HIBA_SPLITS_DIR,
    DEFAULT_PAD_SPLITS_DIR,
    SOURCE_ORDER,
    load_multi_source_manifests,
)
from ml.training.run_pad_hiba_convnext_final_fit import (
    EXPECTED_MANIFEST_IDENTITY_FINGERPRINT,
    manifest_identity_fingerprint,
)
from ml.training.train import get_device, resolve_project_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pad_hiba_open_images_supported_input_gate_v1"
SEED = 42
CALIBRATION_RETENTION_TARGET = 0.95
EVALUATION_RETENTION_MINIMUM = 0.90
EVALUATION_FALSE_ACCEPT_MAXIMUM = 0.05
EVALUATION_GROUP_FALSE_ACCEPT_MAXIMUM = 0.10
METHOD_PREFERENCE = ("logsumexp", "max_logit", "maximum_softmax_probability")
EXPECTED_POSITIVE_RAW_COUNTS = {"pad_ufes": 2_298, "hiba": 309}
EXPECTED_CHECKPOINT_SHA256 = "12c7261b06e3da9d1639e5e2c11220837de5a69f972acf25a55c4a0ae31d99b8"


@dataclass(frozen=True)
class InputRecord:
    path: Path
    partition: str
    kind: str
    group: str
    identity: str


class ScoringDataset(Dataset):
    def __init__(self, records: Sequence[InputRecord], indices: Sequence[int]) -> None:
        self.records = records
        self.indices = list(indices)
        self.transform = get_transforms("val")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        index = self.indices[item]
        with Image.open(self.records[index].path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen obvious-non-skin supported-input gate."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--pad-splits-dir", type=Path, default=DEFAULT_PAD_SPLITS_DIR)
    parser.add_argument("--hiba-splits-dir", type=Path, default=DEFAULT_HIBA_SPLITS_DIR)
    parser.add_argument("--negative-manifest", type=Path, required=True)
    parser.add_argument("--negative-summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _portable_basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _development_partition(identity: str) -> str:
    digest = hashlib.sha256(f"{SEED}:{identity}".encode()).digest()
    return "calibration" if int.from_bytes(digest[:8], "big") < 2**63 else "evaluation"


def load_positive_records(pad_splits_dir: Path, hiba_splits_dir: Path) -> list[InputRecord]:
    manifests = load_multi_source_manifests(pad_splits_dir, hiba_splits_dir)
    observed_fingerprint = manifest_identity_fingerprint(manifests.folds)
    if observed_fingerprint != EXPECTED_MANIFEST_IDENTITY_FINGERPRINT:
        raise ValueError(
            "PAD/HIBA manifest identity drifted: "
            f"expected {EXPECTED_MANIFEST_IDENTITY_FINGERPRINT}, observed {observed_fingerprint}."
        )
    rows = manifests.folds[0]
    raw_counts = {source: int((rows["source"] == source).sum()) for source in SOURCE_ORDER}
    if raw_counts != EXPECTED_POSITIVE_RAW_COUNTS:
        raise ValueError(f"PAD/HIBA positive counts drifted: {raw_counts!r}.")

    records = []
    for row in rows.itertuples(index=False):
        source = str(row.source)
        image_key = _portable_basename(str(row.image_path))
        unit_key = image_key if source == "pad_ufes" else str(row.unit_id)
        identity = f"{source}:{unit_key}"
        records.append(
            InputRecord(
                path=resolve_project_path(Path(str(row.image_path))),
                partition=_development_partition(identity),
                kind="positive",
                group=source,
                identity=identity,
            )
        )
    if len({record.identity + ":" + record.path.name for record in records}) != len(records):
        raise ValueError("PAD/HIBA positive records contain duplicate image identities.")
    return records


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_columns = {
            "partition",
            "semantic_group",
            "class_name",
            "image_id",
            "image_path",
            "license",
            "author",
            "title",
            "sha256",
        }
        if set(reader.fieldnames or ()) != expected_columns:
            raise ValueError("Open Images attribution manifest columns drifted.")
        return list(reader)


def load_negative_records(manifest_path: Path, summary_path: Path) -> list[InputRecord]:
    manifest_path = manifest_path.resolve()
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _read_manifest_rows(manifest_path)
    if not EXPECTED_COHORT_FINGERPRINT:
        raise ValueError("The Open Images cohort fingerprint has not been frozen.")
    expected_summary = {
        "dataset": "Open Images V5",
        "dataset_role": "obvious_non_skin_ood_development",
        "protocol": NEGATIVE_COHORT_PROTOCOL,
        "seed": NEGATIVE_COHORT_SEED,
        "license_required_per_image": EXPECTED_LICENSE,
        "images_per_category": IMAGES_PER_CATEGORY,
        "minimum_target_box_area": MIN_TARGET_BOX_AREA,
        "excluded_human_classes": list(EXCLUDED_HUMAN_CLASSES),
        "category_plan": json.loads(json.dumps(CATEGORY_PLAN)),
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "cohort_fingerprint": EXPECTED_COHORT_FINGERPRINT,
    }
    mismatches = [key for key, expected in expected_summary.items() if summary.get(key) != expected]
    if mismatches:
        raise ValueError("Open Images cohort summary drifted: " + ", ".join(mismatches))
    if summary.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Open Images attribution manifest checksum drifted.")
    if summary.get("cohort_fingerprint") != cohort_fingerprint(rows):
        raise ValueError("Open Images cohort fingerprint drifted.")
    expected_category_counts = {
        (partition, semantic_group, class_name): IMAGES_PER_CATEGORY
        for partition, groups in CATEGORY_PLAN.items()
        for semantic_group, class_names in groups.items()
        for class_name in class_names
    }
    observed_category_counts = Counter(
        (row["partition"], row["semantic_group"], row["class_name"]) for row in rows
    )
    if observed_category_counts != expected_category_counts:
        raise ValueError("Open Images OOD category counts drifted.")

    records = []
    for row in rows:
        if row["license"] != EXPECTED_LICENSE:
            raise ValueError("Open Images manifest contains an unapproved image license.")
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            raise ValueError("Open Images manifest paths must be absolute private paths.")
        image_path = image_path.resolve()
        if not image_path.is_file() or sha256_file(image_path) != row["sha256"]:
            raise ValueError("Open Images cohort image checksum drifted.")
        records.append(
            InputRecord(
                path=image_path,
                partition=row["partition"],
                kind="negative",
                group=row["semantic_group"],
                identity=f"open_images:{row['image_id']}",
            )
        )
    if len({record.identity for record in records}) != len(records):
        raise ValueError("Open Images OOD cohort contains duplicate image identities.")
    return records


def assess_quality(records: Sequence[InputRecord]) -> np.ndarray:
    passed = np.ones(len(records), dtype=bool)
    for index, record in enumerate(records):
        try:
            with Image.open(record.path) as image:
                image.load()
                validate_image_quality(image)
        except PoorImageQualityError:
            passed[index] = False
    return passed


def score_records(
    model: torch.nn.Module,
    records: Sequence[InputRecord],
    quality_passed: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, np.ndarray]:
    passing_indices = np.flatnonzero(quality_passed).tolist()
    dataset = ScoringDataset(records, passing_indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    scores = {
        method: np.full(len(records), -np.inf, dtype=np.float64) for method in METHOD_PREFERENCE
    }
    with torch.inference_mode():
        for tensors, indices in loader:
            logits = model(tensors.to(device))
            probabilities = torch.softmax(logits, dim=1)
            batch_scores = {
                "maximum_softmax_probability": probabilities.max(dim=1).values,
                "max_logit": logits.max(dim=1).values,
                "logsumexp": torch.logsumexp(logits, dim=1),
            }
            target_indices = indices.numpy()
            for method, values in batch_scores.items():
                scores[method][target_indices] = values.detach().cpu().numpy()
    return scores


def _mask(
    records: Sequence[InputRecord],
    *,
    partition: str,
    kind: str,
    group: str | None = None,
) -> np.ndarray:
    return np.asarray(
        [
            record.partition == partition
            and record.kind == kind
            and (group is None or record.group == group)
            for record in records
        ],
        dtype=bool,
    )


def retention(decisions: np.ndarray, mask: np.ndarray) -> float:
    count = int(mask.sum())
    if count == 0:
        raise ValueError("Cannot calculate retention for an empty group.")
    return float(decisions[mask].mean())


def choose_strictest_threshold(
    score: np.ndarray,
    quality_passed: np.ndarray,
    positive_masks: Sequence[np.ndarray],
    *,
    minimum_retention: float,
) -> float | None:
    calibration_positive_mask = np.logical_or.reduce(positive_masks)
    candidates = np.unique(score[calibration_positive_mask & np.isfinite(score)])
    for threshold in candidates[::-1]:
        decisions = quality_passed & (score >= threshold)
        if all(retention(decisions, mask) >= minimum_retention for mask in positive_masks):
            return float(threshold)
    return None


def grouped_rates(
    decisions: np.ndarray,
    records: Sequence[InputRecord],
    *,
    partition: str,
    kind: str,
) -> dict[str, float]:
    groups = sorted(
        {
            record.group
            for record in records
            if record.partition == partition and record.kind == kind
        }
    )
    return {
        group: retention(
            decisions,
            _mask(records, partition=partition, kind=kind, group=group),
        )
        for group in groups
    }


def count_groups(records: Sequence[InputRecord], *, partition: str, kind: str) -> dict[str, int]:
    groups = sorted(
        {
            record.group
            for record in records
            if record.partition == partition and record.kind == kind
        }
    )
    return {
        group: int(_mask(records, partition=partition, kind=kind, group=group).sum())
        for group in groups
    }


def select_gate(
    records: Sequence[InputRecord],
    quality_passed: np.ndarray,
    scores: Mapping[str, np.ndarray],
) -> tuple[str, float, dict[str, dict[str, object]]]:
    positive_masks = [
        _mask(records, partition="calibration", kind="positive", group=source)
        for source in SOURCE_ORDER
    ]
    negative_mask = _mask(records, partition="calibration", kind="negative")
    candidates: dict[str, dict[str, object]] = {}
    for method in METHOD_PREFERENCE:
        threshold = choose_strictest_threshold(
            scores[method],
            quality_passed,
            positive_masks,
            minimum_retention=CALIBRATION_RETENTION_TARGET,
        )
        if threshold is None:
            candidates[method] = {"eligible": False}
            continue
        decisions = quality_passed & (scores[method] >= threshold)
        positive_retention = {
            source: retention(
                decisions,
                _mask(records, partition="calibration", kind="positive", group=source),
            )
            for source in SOURCE_ORDER
        }
        false_accept_by_group = grouped_rates(
            decisions, records, partition="calibration", kind="negative"
        )
        candidates[method] = {
            "eligible": True,
            "threshold": threshold,
            "positive_retention": positive_retention,
            "negative_false_accept_rate": retention(decisions, negative_mask),
            "negative_false_accept_rate_by_group": false_accept_by_group,
        }
    eligible = [method for method in METHOD_PREFERENCE if candidates[method]["eligible"]]
    if not eligible:
        raise ValueError("No supported-input score can retain the calibration target.")
    preference = {method: index for index, method in enumerate(METHOD_PREFERENCE)}
    selected = min(
        eligible,
        key=lambda method: (
            float(candidates[method]["negative_false_accept_rate"]),
            preference[method],
        ),
    )
    return selected, float(candidates[selected]["threshold"]), candidates


def _quality_counts(
    records: Sequence[InputRecord], quality_passed: np.ndarray, *, partition: str, kind: str
) -> dict[str, int]:
    return {
        group: int(
            (~quality_passed & _mask(records, partition=partition, kind=kind, group=group)).sum()
        )
        for group in count_groups(records, partition=partition, kind=kind)
    }


def evaluate_gate(args: argparse.Namespace) -> dict[str, object]:
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers must be non-negative.")
    checkpoint_path = resolve_project_path(args.checkpoint).resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "Supported-input evaluation checkpoint drifted: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, observed {checkpoint_sha256}."
        )
    negative_manifest = resolve_project_path(args.negative_manifest).resolve()
    negative_summary = resolve_project_path(args.negative_summary).resolve()
    records = load_positive_records(args.pad_splits_dir, args.hiba_splits_dir)
    records.extend(load_negative_records(negative_manifest, negative_summary))
    quality_passed = assess_quality(records)
    device = get_device(args.device)
    model = load_model(checkpoint_path, device)
    scores = score_records(
        model,
        records,
        quality_passed,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    selected_method, threshold, calibration_candidates = select_gate(
        records, quality_passed, scores
    )

    evaluation_decisions = quality_passed & (scores[selected_method] >= threshold)
    positive_retention = {
        source: retention(
            evaluation_decisions,
            _mask(records, partition="evaluation", kind="positive", group=source),
        )
        for source in SOURCE_ORDER
    }
    negative_mask = _mask(records, partition="evaluation", kind="negative")
    negative_false_accept_rate = retention(evaluation_decisions, negative_mask)
    negative_false_accept_by_group = grouped_rates(
        evaluation_decisions, records, partition="evaluation", kind="negative"
    )
    decision_rules = {
        "pad_evaluation_retention_gte_0_90": positive_retention["pad_ufes"]
        >= EVALUATION_RETENTION_MINIMUM,
        "hiba_evaluation_retention_gte_0_90": positive_retention["hiba"]
        >= EVALUATION_RETENTION_MINIMUM,
        "obvious_non_skin_false_accept_lte_0_05": negative_false_accept_rate
        <= EVALUATION_FALSE_ACCEPT_MAXIMUM,
        "every_semantic_group_false_accept_lte_0_10": all(
            value <= EVALUATION_GROUP_FALSE_ACCEPT_MAXIMUM
            for value in negative_false_accept_by_group.values()
        ),
    }
    decision_rules["all_pass"] = all(decision_rules.values())
    negative_summary_data = json.loads(negative_summary.read_text(encoding="utf-8"))
    report = {
        "context": (
            "Development-only supported-input evaluation for obvious non-skin images; not "
            "diagnosis, independent classifier validation, or evidence for unknown conditions."
        ),
        "protocol": PROTOCOL,
        "seed": SEED,
        "checkpoint_sha256": checkpoint_sha256,
        "positive_manifest_identity_fingerprint": EXPECTED_MANIFEST_IDENTITY_FINGERPRINT,
        "negative_cohort_fingerprint": negative_summary_data["cohort_fingerprint"],
        "negative_manifest_sha256": negative_summary_data["manifest_sha256"],
        "negative_source_sha256": negative_summary_data["source_sha256"],
        "score_candidates": list(METHOD_PREFERENCE),
        "calibration_retention_target": CALIBRATION_RETENTION_TARGET,
        "selection": {
            "method": selected_method,
            "threshold": threshold,
            "candidates": calibration_candidates,
        },
        "counts": {
            partition: {
                "positive": count_groups(records, partition=partition, kind="positive"),
                "negative": count_groups(records, partition=partition, kind="negative"),
            }
            for partition in ("calibration", "evaluation")
        },
        "quality_rejections": {
            partition: {
                "positive": _quality_counts(
                    records, quality_passed, partition=partition, kind="positive"
                ),
                "negative": _quality_counts(
                    records, quality_passed, partition=partition, kind="negative"
                ),
            }
            for partition in ("calibration", "evaluation")
        },
        "evaluation": {
            "positive_retention": positive_retention,
            "obvious_non_skin_false_accept_rate": negative_false_accept_rate,
            "obvious_non_skin_false_accept_rate_by_group": negative_false_accept_by_group,
        },
        "decision_rules": decision_rules,
        "independent_classifier_evaluation": False,
        "milk10k_used": False,
        "privacy": {
            "aggregate_metrics_only": True,
            "per_image_scores_written": False,
            "per_image_decisions_written": False,
            "identifiers_or_paths_written": False,
        },
        "limitations": (
            "The final classifier saw the positive development images. The negative evaluation "
            "uses fixed obvious non-skin Open Images categories and does not establish detection "
            "of unsupported skin conditions, arbitrary inputs, or consumer-photo reliability."
        ),
    }
    if not all(math.isfinite(float(value)) for value in (threshold, negative_false_accept_rate)):
        raise ValueError("Supported-input report contains a non-finite selected result.")
    report_path = resolve_project_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"Supported-input gate decision={'enable' if decision_rules['all_pass'] else 'reject'} "
        f"method={selected_method} report={report_path.name}"
    )
    return report


def main() -> None:
    evaluate_gate(parse_args())


if __name__ == "__main__":
    main()
