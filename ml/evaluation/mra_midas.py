"""Evaluate frozen PAD-UFES-native checkpoints on authorized MRA-MIDAS data.

The primary protocol is a six-class, lesion-level evaluation over paired 15-cm
and 30-cm standardized iPhone/iPad clinical photographs. Dermoscopy,
portal-submitted virtual photographs, unbiopsied controls, ambiguous filename
mappings, and pathology categories outside CLEAR's PAD-UFES-native taxonomy
are not scored.

Outputs contain aggregate experimental metrics only. They are not medical
conclusions and must remain on ignored paths.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.evaluation.metrics import summarize_metrics
from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.preprocessing import get_pad_ufes_transforms
from ml.training.train_pad_ufes import ARCHITECTURES, build_transfer_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "mra_midas"
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "ml" / "models" / "pad_ufes_convnext_tiny_cv_seed42"
DEFAULT_OUT_DIR = PROJECT_ROOT / "ml" / "runs" / "evaluation" / "mra-midas-convnext-tiny-cv"
DEFAULT_FOLDS = 5
DEFAULT_SEED = 42
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
EXPECTED_PREPROCESSING = "resize_224_imagenet_normalization"
CLINICAL_DISTANCES = ("6in", "1ft")
VIRTUAL_DISTANCE = "n/a - virtual"
DERMOSCOPY_DISTANCE = "dscope"

EXPECTED_SOURCE_HASHES = {
    "release_midas.csv": "176eeaf906785eb80dd870171b38425467aa307e271e3529d616d70124882ff4",
    "data.csv": "07f9b6ee213b519daa6a287fea5e4b4d72f0b3d56e281e8f533768ff946454e9",
}
REQUIRED_RELEASE_COLUMNS = {
    "_unnamed_var",
    "midas_record_id",
    "midas_file_name",
    "midas_iscontrol",
    "midas_distance",
    "midas_location",
    "midas_path",
    "midas_pathreport",
    "midas_gender",
    "midas_age",
    "midas_fitzpatrick",
    "midas_melanoma",
    "midas_ethnicity",
    "midas_race",
    "clinical_impression_1",
    "clinical_impression_2",
    "clinical_impression_3",
    "length__mm_",
    "width__mm_",
}
REQUIRED_MANIFEST_COLUMNS = {"file_id", "file_name", "size", "added_at", "md5_hash"}
PROFILE_COLUMNS = (
    "midas_record_id",
    "midas_iscontrol",
    "midas_location",
    "midas_path",
    "midas_pathreport",
    "clinical_impression_1",
    "clinical_impression_2",
    "clinical_impression_3",
    "length__mm_",
    "width__mm_",
)
PATHOLOGY_TO_PAD_UFES = {
    "malignant- ak": "actinic_keratosis",
    "malignant- bcc": "basal_cell_carcinoma",
    "malignant- melanoma": "melanoma",
    "benign-melanocytic nevus": "nevus",
    "malignant- scc": "squamous_cell_carcinoma",
    "benign-seborrheic keratosis": "seborrheic_keratosis",
}
MRA_MIDAS_CAPTURE_CONTEXT = {
    "source": "prospectively_recruited_stanford_clinical_lesions",
    "clinical_devices": "contemporary_iphone_or_ipad",
    "clinical_distances": ["15_cm", "30_cm"],
    "flash": False,
    "capture_setting": "standardized_research_clinic_photography",
    "photographer": "not_identified_in_public_source",
    "standardized_iphone_ipad_clinical_capture_validation": True,
    "patient_taken_validation": False,
    "virtual_source": "patient_or_primary_care_provider_portal_submission",
    "virtual_images_scored": False,
    "dermoscopy_scored": False,
}
WORKFLOW_ARTIFACT_CONTEXT = {
    "audit_status": "not_performed",
    "not_established": [
        "ruler_prevalence",
        "skin_marker_prevalence",
        "procedure_related_artifact_prevalence",
    ],
}
INTERVAL_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")

EXPECTED_FROZEN_AUDIT = {
    "source_row_count": 3_416,
    "manifest_row_count": 3_416,
    "derived_lesion_profile_count": 1_192,
    "control_profile_count": 226,
    "non_control_profile_count": 966,
    "exact_six_class_profile_count": 682,
    "primary_paired_lesion_count": 667,
    "primary_record_count": 464,
    "primary_repeated_view_profile_count": 12,
    "primary_image_support": {"6in": 677, "1ft": 672},
    "primary_class_support": {
        "actinic_keratosis": 62,
        "basal_cell_carcinoma": 199,
        "melanoma": 74,
        "nevus": 188,
        "squamous_cell_carcinoma": 66,
        "seborrheic_keratosis": 78,
    },
    "filename_resolution": {
        "exact": 3_155,
        "unique_stem": 203,
        "mutual_unique_normalized_prefix": 45,
        "quarantined": 13,
    },
    "quarantined_by_distance": {"1ft": 4, "6in": 3, "n/a - virtual": 6},
    "primary_fitzpatrick_support": {
        "FST_I_II": 601,
        "FST_III_IV": 61,
        "FST_V_VI": 2,
        "unknown_or_inconsistent": 3,
    },
}


@dataclass(frozen=True)
class PreparedMraMidas:
    release_rows: pd.DataFrame
    manifest_rows: pd.DataFrame
    resolved_rows: pd.DataFrame
    image_rows: pd.DataFrame
    lesion_rows: pd.DataFrame
    audit: dict[str, object]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen PAD-UFES-native checkpoints on Stanford MRA-MIDAS."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="convnext_tiny")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def resolve_project_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_authorized_source_hashes(data_dir: Path) -> dict[str, str]:
    observed = {}
    for filename, expected in EXPECTED_SOURCE_HASHES.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing MRA-MIDAS source table: {path}")
        observed[filename] = _sha256(path)
        if observed[filename] != expected:
            raise ValueError(f"Authorized MRA-MIDAS source hash mismatch: {filename}")
    return observed


def load_source_tables(
    data_dir: Path,
    *,
    validate_files: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the authorized tables and validate the complete extracted manifest."""
    data_dir = resolve_project_path(data_dir)
    release_path = data_dir / "release_midas.csv"
    manifest_path = data_dir / "data.csv"
    images_dir = data_dir / "images"
    if not release_path.is_file():
        raise FileNotFoundError(f"Missing MRA-MIDAS metadata: {release_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing MRA-MIDAS file manifest: {manifest_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing MRA-MIDAS images directory: {images_dir}")

    release_rows = pd.read_csv(release_path, dtype=str, keep_default_na=False)
    manifest_rows = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    _validate_columns(release_rows, REQUIRED_RELEASE_COLUMNS, name="scientific metadata")
    _validate_columns(manifest_rows, REQUIRED_MANIFEST_COLUMNS, name="file manifest")
    if release_rows.empty or manifest_rows.empty:
        raise ValueError("MRA-MIDAS source tables must not be empty.")

    for column in ("midas_record_id", "midas_file_name", "midas_distance"):
        release_rows[column] = release_rows[column].str.strip()
        if bool((release_rows[column] == "").any()):
            raise ValueError(f"MRA-MIDAS metadata contains blank {column} values.")
    for column in ("file_id", "file_name", "size", "md5_hash"):
        manifest_rows[column] = manifest_rows[column].str.strip()
        if bool((manifest_rows[column] == "").any()):
            raise ValueError(f"MRA-MIDAS manifest contains blank {column} values.")

    if bool(release_rows["midas_file_name"].str.lower().duplicated().any()):
        raise ValueError("MRA-MIDAS metadata contains duplicate filenames.")
    if bool(manifest_rows["file_id"].duplicated().any()):
        raise ValueError("MRA-MIDAS manifest contains duplicate file_id values.")
    if bool(manifest_rows["file_name"].str.lower().duplicated().any()):
        raise ValueError("MRA-MIDAS manifest contains duplicate filenames.")

    allowed_distances = {*CLINICAL_DISTANCES, VIRTUAL_DISTANCE, DERMOSCOPY_DISTANCE}
    unexpected_distances = sorted(set(release_rows["midas_distance"]) - allowed_distances)
    if unexpected_distances:
        raise ValueError(f"MRA-MIDAS metadata contains unknown distances: {unexpected_distances}")
    unexpected_controls = sorted(set(release_rows["midas_iscontrol"]) - {"yes", "no"})
    if unexpected_controls:
        raise ValueError(
            f"MRA-MIDAS metadata contains invalid control values: {unexpected_controls}"
        )

    release_rows = release_rows.reset_index(drop=True)
    release_rows["source_row_index"] = np.arange(len(release_rows), dtype=int)
    manifest_rows = _validate_manifest_rows(
        manifest_rows,
        images_dir,
        validate_files=validate_files,
    )
    return release_rows, manifest_rows


def _validate_columns(rows: pd.DataFrame, expected: set[str], *, name: str) -> None:
    missing = expected.difference(rows.columns)
    if missing:
        raise ValueError(f"MRA-MIDAS {name} is missing columns: {', '.join(sorted(missing))}")


def _safe_relative_manifest_name(value: str) -> PurePosixPath:
    if "\\" in value:
        raise ValueError("MRA-MIDAS manifest paths must use forward slashes.")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("MRA-MIDAS manifest contains an unsafe file path.")
    return path


def _validate_manifest_rows(
    rows: pd.DataFrame,
    images_dir: Path,
    *,
    validate_files: bool,
) -> pd.DataFrame:
    root = images_dir.resolve()
    normalized_names = []
    paths = []
    sizes = []
    decoded_md5 = []
    for row in rows.itertuples(index=False):
        relative = _safe_relative_manifest_name(str(row.file_name))
        path = (images_dir / Path(*relative.parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("MRA-MIDAS manifest path escapes the image directory.") from exc
        try:
            size = int(row.size)
        except ValueError as exc:
            raise ValueError("MRA-MIDAS manifest contains a non-integer size.") from exc
        if size <= 0:
            raise ValueError("MRA-MIDAS manifest sizes must be positive.")
        try:
            digest = base64.b64decode(str(row.md5_hash), validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError("MRA-MIDAS manifest contains an invalid Base64 MD5 digest.") from exc
        if len(digest) != hashlib.md5().digest_size:
            raise ValueError("MRA-MIDAS manifest MD5 digests must contain 16 bytes.")
        normalized_names.append(relative.as_posix())
        paths.append(path)
        sizes.append(size)
        decoded_md5.append(digest)

    rows = rows.copy()
    rows["file_name"] = normalized_names
    rows["image_path"] = paths
    rows["size_bytes"] = sizes
    rows["md5_bytes"] = decoded_md5
    if not validate_files:
        return rows

    expected_names = set(normalized_names)
    actual_paths = [path for path in images_dir.rglob("*") if path.is_file()]
    actual_names = {path.relative_to(images_dir).as_posix() for path in actual_paths}
    if expected_names != actual_names:
        raise ValueError(
            "MRA-MIDAS image files do not match the manifest: "
            f"missing={len(expected_names - actual_names)} "
            f"extra={len(actual_names - expected_names)}."
        )

    for row in rows.itertuples(index=False):
        path = Path(row.image_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("MRA-MIDAS manifest contains a missing or symbolic-link file.")
        if path.stat().st_size != row.size_bytes:
            raise ValueError("MRA-MIDAS image size does not match the manifest.")
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.digest() != row.md5_bytes:
            raise ValueError("MRA-MIDAS image MD5 does not match the manifest.")
    return rows


def reconcile_filenames(release_rows: pd.DataFrame, manifest_rows: pd.DataFrame) -> pd.DataFrame:
    """Resolve scientific filenames to archive files without ambiguous matching."""
    actual_by_basename = {}
    for row in manifest_rows.itertuples(index=False):
        basename = PurePosixPath(row.file_name).name.lower()
        if basename in actual_by_basename:
            raise ValueError("MRA-MIDAS manifest basenames must be unique.")
        actual_by_basename[basename] = row

    resolved: dict[int, tuple[Path, str, str]] = {}
    used_actual: set[str] = set()
    for row in release_rows.itertuples(index=False):
        scientific_name = str(row.midas_file_name)
        if (
            Path(scientific_name).name != scientific_name
            or "/" in scientific_name
            or "\\" in scientific_name
        ):
            raise ValueError("MRA-MIDAS scientific filenames must be plain filenames.")
        key = scientific_name.lower()
        actual = actual_by_basename.get(key)
        if actual is not None:
            resolved[row.source_row_index] = (Path(actual.image_path), actual.file_name, "exact")
            used_actual.add(key)

    unresolved_rows = release_rows[~release_rows["source_row_index"].isin(resolved)]
    unused_actual = {
        key: value for key, value in actual_by_basename.items() if key not in used_actual
    }
    _resolve_unique_stems(unresolved_rows, unused_actual, resolved, used_actual)
    unresolved_rows = release_rows[~release_rows["source_row_index"].isin(resolved)]
    unused_actual = {
        key: value for key, value in actual_by_basename.items() if key not in used_actual
    }
    _resolve_mutual_unique_prefixes(unresolved_rows, unused_actual, resolved, used_actual)

    result = release_rows.copy()
    result["filename_resolution"] = "quarantined"
    result["manifest_file_name"] = ""
    result["image_path"] = None
    for index, (path, manifest_name, resolution) in resolved.items():
        mask = result["source_row_index"] == index
        result.loc[mask, "filename_resolution"] = resolution
        result.loc[mask, "manifest_file_name"] = manifest_name
        result.loc[mask, "image_path"] = str(path)

    resolved_paths = result.loc[result["filename_resolution"] != "quarantined", "image_path"]
    if bool(resolved_paths.duplicated().any()):
        raise ValueError("MRA-MIDAS filename reconciliation is not one-to-one.")
    if len(resolved) + int((result["filename_resolution"] == "quarantined").sum()) != len(result):
        raise RuntimeError("MRA-MIDAS filename reconciliation lost source rows.")
    return result


def _filename_stem(value: str) -> str:
    return Path(value).stem.lower()


def _normalized_stem(value: str) -> str:
    return "".join(character for character in _filename_stem(value) if character.isalnum())


def _resolve_unique_stems(
    rows: pd.DataFrame,
    actual: Mapping[str, object],
    resolved: dict[int, tuple[Path, str, str]],
    used_actual: set[str],
) -> None:
    source_by_stem: defaultdict[str, list[object]] = defaultdict(list)
    actual_by_stem: defaultdict[str, list[tuple[str, object]]] = defaultdict(list)
    for row in rows.itertuples(index=False):
        source_by_stem[_filename_stem(row.midas_file_name)].append(row)
    for key, row in actual.items():
        actual_by_stem[_filename_stem(key)].append((key, row))
    for stem, source_matches in source_by_stem.items():
        actual_matches = actual_by_stem.get(stem, [])
        if len(source_matches) == 1 and len(actual_matches) == 1:
            source = source_matches[0]
            key, archive = actual_matches[0]
            resolved[source.source_row_index] = (
                Path(archive.image_path),
                archive.file_name,
                "unique_stem",
            )
            used_actual.add(key)


def _resolve_mutual_unique_prefixes(
    rows: pd.DataFrame,
    actual: Mapping[str, object],
    resolved: dict[int, tuple[Path, str, str]],
    used_actual: set[str],
) -> None:
    source_rows = list(rows.itertuples(index=False))
    actual_rows = list(actual.items())
    candidates: list[tuple[object, str, object]] = []
    for source in source_rows:
        source_norm = _normalized_stem(source.midas_file_name)
        if not source_norm:
            continue
        matches = [
            (key, archive)
            for key, archive in actual_rows
            if _is_prefix_pair(source_norm, _normalized_stem(key))
        ]
        if len(matches) != 1:
            continue
        key, archive = matches[0]
        archive_norm = _normalized_stem(key)
        reverse = [
            other
            for other in source_rows
            if _is_prefix_pair(_normalized_stem(other.midas_file_name), archive_norm)
        ]
        if len(reverse) == 1:
            candidates.append((source, key, archive))

    candidate_actual = [key for _, key, _ in candidates]
    if len(candidate_actual) != len(set(candidate_actual)):
        raise ValueError("MRA-MIDAS prefix reconciliation produced duplicate candidates.")
    for source, key, archive in candidates:
        resolved[source.source_row_index] = (
            Path(archive.image_path),
            archive.file_name,
            "mutual_unique_normalized_prefix",
        )
        used_actual.add(key)


def _is_prefix_pair(left: str, right: str) -> bool:
    return bool(left and right) and (left.startswith(right) or right.startswith(left))


def prepare_primary_cohort(
    release_rows: pd.DataFrame,
    manifest_rows: pd.DataFrame,
) -> PreparedMraMidas:
    resolved_rows = reconcile_filenames(release_rows, manifest_rows)
    grouped = list(resolved_rows.groupby(list(PROFILE_COLUMNS), sort=True, dropna=False))
    image_records = []
    lesion_records = []
    profile_summaries = []
    for profile_key, group in grouped:
        profile_id = hashlib.sha256(
            json.dumps(profile_key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        first = group.iloc[0]
        label = PATHOLOGY_TO_PAD_UFES.get(first["midas_path"].strip())
        is_control = first["midas_iscontrol"] == "yes"
        resolved_group = group[group["filename_resolution"] != "quarantined"]
        clinical = {
            distance: resolved_group[resolved_group["midas_distance"] == distance]
            for distance in CLINICAL_DISTANCES
        }
        virtual = resolved_group[resolved_group["midas_distance"] == VIRTUAL_DISTANCE]
        profile_summaries.append(
            {
                "profile_id": profile_id,
                "is_control": is_control,
                "pathology": first["midas_path"].strip(),
                "label": label,
                "has_pathology": bool(first["midas_path"].strip()),
                "resolved_6in": len(clinical["6in"]),
                "resolved_1ft": len(clinical["1ft"]),
                "resolved_virtual": len(virtual),
                "fst_group": _fitzpatrick_group(group["midas_fitzpatrick"]),
                "gender_value_count": int(group["midas_gender"].nunique()),
            }
        )
        missing_required_view = any(clinical[distance].empty for distance in CLINICAL_DISTANCES)
        if is_control or label is None or missing_required_view:
            continue
        lesion_index = len(lesion_records)
        lesion_records.append(
            {
                "lesion_index": lesion_index,
                "profile_id": profile_id,
                "record_id": first["midas_record_id"],
                "label": label,
                "fst_group": _fitzpatrick_group(group["midas_fitzpatrick"]),
                "image_count_6in": len(clinical["6in"]),
                "image_count_1ft": len(clinical["1ft"]),
            }
        )
        for distance in CLINICAL_DISTANCES:
            for row in clinical[distance].itertuples(index=False):
                image_records.append(
                    {
                        "image_index": len(image_records),
                        "lesion_index": lesion_index,
                        "distance": distance,
                        "image_path": Path(row.image_path),
                    }
                )

    image_rows = pd.DataFrame(image_records)
    lesion_rows = pd.DataFrame(lesion_records)
    if image_rows.empty or lesion_rows.empty:
        raise ValueError("MRA-MIDAS primary paired-view cohort is empty.")
    image_rows = image_rows.sort_values("image_index").reset_index(drop=True)
    lesion_rows = lesion_rows.sort_values("lesion_index").reset_index(drop=True)
    audit = build_dataset_audit(
        release_rows,
        manifest_rows,
        resolved_rows,
        pd.DataFrame(profile_summaries),
        image_rows,
        lesion_rows,
    )
    return PreparedMraMidas(
        release_rows=release_rows,
        manifest_rows=manifest_rows,
        resolved_rows=resolved_rows,
        image_rows=image_rows,
        lesion_rows=lesion_rows,
        audit=audit,
    )


def _fitzpatrick_group(values: pd.Series) -> str:
    normalized = {str(value).strip().lower() for value in values if str(value).strip()}
    if len(normalized) != 1:
        return "unknown_or_inconsistent"
    roman = next(iter(normalized)).split(maxsplit=1)[0]
    if roman in {"i", "ii"}:
        return "FST_I_II"
    if roman in {"iii", "iv"}:
        return "FST_III_IV"
    if roman in {"v", "vi"}:
        return "FST_V_VI"
    return "unknown_or_inconsistent"


def build_dataset_audit(
    release_rows: pd.DataFrame,
    manifest_rows: pd.DataFrame,
    resolved_rows: pd.DataFrame,
    profiles: pd.DataFrame,
    image_rows: pd.DataFrame,
    lesion_rows: pd.DataFrame,
) -> dict[str, object]:
    resolution_counts = Counter(resolved_rows["filename_resolution"])
    quarantined = resolved_rows[resolved_rows["filename_resolution"] == "quarantined"]
    primary_class_support = Counter(lesion_rows["label"])
    primary_fst_support = Counter(lesion_rows["fst_group"])
    exact_profiles = profiles[profiles["label"].notna() & ~profiles["is_control"]]
    virtual_exact = exact_profiles[exact_profiles["resolved_virtual"] > 0]
    return {
        "dataset": "Stanford MRA-MIDAS",
        "dataset_doi": "10.71718/15nz-jv40",
        "study_doi": "10.1056/aidbp2400732",
        "source_row_count": len(release_rows),
        "manifest_row_count": len(manifest_rows),
        "manifest_unique_file_count": int(manifest_rows["file_name"].nunique()),
        "manifest_total_bytes": int(manifest_rows["size_bytes"].sum()),
        "record_id_count": int(release_rows["midas_record_id"].nunique()),
        "derived_lesion_profile_count": len(profiles),
        "control_profile_count": int(profiles["is_control"].sum()),
        "non_control_profile_count": int((~profiles["is_control"]).sum()),
        "exact_six_class_profile_count": len(exact_profiles),
        "filename_resolution": {
            name: int(resolution_counts.get(name, 0))
            for name in (
                "exact",
                "unique_stem",
                "mutual_unique_normalized_prefix",
                "quarantined",
            )
        },
        "quarantined_by_distance": {
            key: int(value) for key, value in sorted(Counter(quarantined["midas_distance"]).items())
        },
        "pathology_mapping": dict(PATHOLOGY_TO_PAD_UFES),
        "scc_in_situ_mapping": "excluded_not_folded_into_squamous_cell_carcinoma",
        "primary_paired_lesion_count": len(lesion_rows),
        "primary_record_count": int(lesion_rows["record_id"].nunique()),
        "primary_class_support": {
            label: int(primary_class_support.get(label, 0)) for label in PAD_UFES_NATIVE_LABELS
        },
        "primary_image_support": {
            distance: int((image_rows["distance"] == distance).sum())
            for distance in CLINICAL_DISTANCES
        },
        "primary_repeated_view_profile_count": int(
            ((lesion_rows["image_count_6in"] > 1) | (lesion_rows["image_count_1ft"] > 1)).sum()
        ),
        "primary_fitzpatrick_support": {
            group: int(primary_fst_support.get(group, 0))
            for group in (
                "FST_I_II",
                "FST_III_IV",
                "FST_V_VI",
                "unknown_or_inconsistent",
            )
        },
        "gender_data_quality": {
            "profiles_with_conflicting_values": int((profiles["gender_value_count"] > 1).sum()),
            "performance_reporting": "suppressed",
        },
        "virtual_exact_class_support": {
            "profile_count": len(virtual_exact),
            "image_count": int(virtual_exact["resolved_virtual"].sum()),
            "performance_reporting": "withheld_due_to_sparse_incomplete_class_support",
        },
        "capture_context": MRA_MIDAS_CAPTURE_CONTEXT,
        "workflow_artifact_context": WORKFLOW_ARTIFACT_CONTEXT,
        "demographic_performance_reporting": (
            "withheld_due_to_gender_inconsistency_and_skin_tone_race_ethnicity_imbalance"
        ),
        "report_scope": "aggregate_counts_only",
    }


def validate_frozen_audit(audit: Mapping[str, object]) -> None:
    mismatches = []
    for key, expected in EXPECTED_FROZEN_AUDIT.items():
        observed = audit.get(key)
        if observed != expected:
            mismatches.append(f"{key}={observed!r}")
    if mismatches:
        raise ValueError("MRA-MIDAS frozen cohort audit drifted: " + ", ".join(mismatches))


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, object],
    *,
    architecture: str,
    seed: int,
) -> None:
    expected = {
        "architecture": architecture,
        "input_mode": "image_only",
        "dataset": "pad_ufes",
        "label_set": "pad_ufes_native",
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "preprocessing": EXPECTED_PREPROCESSING,
        "seed": seed,
    }
    mismatches = [
        f"{key}={checkpoint.get(key)!r}"
        for key, expected_value in expected.items()
        if checkpoint.get(key) != expected_value
    ]
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        mismatches.append("model_state_dict is missing")
    if mismatches:
        raise ValueError(
            "Checkpoint does not match the frozen PAD-UFES-native protocol: "
            + ", ".join(mismatches)
        )


def discover_checkpoint_paths(checkpoints_dir: Path, *, folds: int) -> list[Path]:
    checkpoints_dir = resolve_project_path(checkpoints_dir)
    if folds < 1:
        raise ValueError("folds must be positive.")
    paths = [checkpoints_dir / f"fold_{fold_index}.pt" for fold_index in range(folds)]
    missing_count = sum(not path.is_file() for path in paths)
    if missing_count:
        raise FileNotFoundError(
            f"Missing {missing_count} of {folds} frozen fold checkpoints in {checkpoints_dir}."
        )
    return paths


class MraMidasImageDataset(Dataset):
    def __init__(self, rows: pd.DataFrame) -> None:
        self.image_paths = [Path(path) for path in rows["image_path"]]
        self.transform = get_pad_ufes_transforms("val")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.image_paths[index]) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, index


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _predict_checkpoint(
    checkpoint_path: Path,
    loader: DataLoader,
    *,
    sample_count: int,
    architecture: str,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {checkpoint_path}")
    validate_checkpoint_metadata(checkpoint, architecture=architecture, seed=seed)

    model = build_transfer_model(architecture=architecture, weights="none")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    probabilities = np.empty((sample_count, len(PAD_UFES_NATIVE_LABELS)), dtype=np.float64)
    seen = np.zeros(sample_count, dtype=bool)
    with torch.inference_mode():
        for images, indices in loader:
            logits = model(images.to(device, non_blocking=True))
            batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            index_array = indices.numpy()
            probabilities[index_array] = batch_probabilities
            seen[index_array] = True
    if not seen.all():
        raise RuntimeError("Inference did not produce probabilities for every MRA-MIDAS image.")

    del model
    del checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probabilities


def aggregate_lesion_probabilities(
    probabilities: np.ndarray,
    image_rows: pd.DataFrame,
    lesion_rows: pd.DataFrame,
) -> dict[str, np.ndarray]:
    array = _validated_probability_matrix(probabilities, expected_rows=len(image_rows))
    by_distance = {
        distance: np.empty((len(lesion_rows), len(PAD_UFES_NATIVE_LABELS)), dtype=np.float64)
        for distance in CLINICAL_DISTANCES
    }
    for lesion in lesion_rows.itertuples(index=False):
        lesion_mask = image_rows["lesion_index"].to_numpy() == lesion.lesion_index
        for distance in CLINICAL_DISTANCES:
            mask = lesion_mask & (image_rows["distance"].to_numpy() == distance)
            if not mask.any():
                raise ValueError(f"MRA-MIDAS lesion is missing the required {distance} view.")
            by_distance[distance][lesion.lesion_index] = array[mask].mean(axis=0)
    combined = np.mean(np.stack([by_distance[distance] for distance in CLINICAL_DISTANCES]), axis=0)
    return {"paired_equal_view_mean": combined, **by_distance}


def _validated_probability_matrix(probabilities: np.ndarray, *, expected_rows: int) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    expected_shape = (expected_rows, len(PAD_UFES_NATIVE_LABELS))
    if array.shape != expected_shape:
        raise ValueError(f"probabilities must have shape {expected_shape}.")
    if not np.isfinite(array).all():
        raise ValueError("probabilities must be finite.")
    if (array < -1e-7).any() or (array > 1.0 + 1e-7).any():
        raise ValueError("probabilities must be between zero and one.")
    if not np.allclose(array.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("each probability row must sum to one.")
    return array


def multiclass_metrics(truth: Sequence[str], probabilities: np.ndarray) -> dict[str, object]:
    truth_values = list(truth)
    array = _validated_probability_matrix(probabilities, expected_rows=len(truth_values))
    predictions = [PAD_UFES_NATIVE_LABELS[index] for index in np.argmax(array, axis=1)]
    return summarize_metrics(truth_values, predictions, labels=PAD_UFES_NATIVE_LABELS)


def record_cluster_bootstrap_intervals(
    truth: Sequence[str],
    probabilities: np.ndarray,
    record_ids: Sequence[str],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    truth_values = np.asarray(list(truth), dtype=object)
    array = _validated_probability_matrix(probabilities, expected_rows=len(truth_values))
    record_values = np.asarray(list(record_ids), dtype=object)
    if record_values.ndim != 1 or len(record_values) != len(truth_values):
        raise ValueError("record_ids must align with truth and probabilities.")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100.")
    if set(truth_values) != set(PAD_UFES_NATIVE_LABELS):
        raise ValueError("record-clustered intervals require all six PAD-UFES labels.")

    predictions = np.asarray(
        [PAD_UFES_NATIVE_LABELS[index] for index in np.argmax(array, axis=1)], dtype=object
    )
    clusters = sorted(set(record_values.tolist()))
    cluster_indices = {cluster: np.flatnonzero(record_values == cluster) for cluster in clusters}
    rng = np.random.default_rng(seed)
    distributions = {metric: [] for metric in INTERVAL_METRICS}
    per_class_recall = {label: [] for label in PAD_UFES_NATIVE_LABELS}
    per_class_f1 = {label: [] for label in PAD_UFES_NATIVE_LABELS}
    for _ in range(samples):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([cluster_indices[cluster] for cluster in selected])
        metrics = summarize_metrics(
            truth_values[indices].tolist(),
            predictions[indices].tolist(),
            labels=PAD_UFES_NATIVE_LABELS,
        )
        for metric in INTERVAL_METRICS:
            distributions[metric].append(float(metrics[metric]))
        for label in PAD_UFES_NATIVE_LABELS:
            per_class_recall[label].append(float(metrics["per_class"][label]["recall"]))
            per_class_f1[label].append(float(metrics["per_class"][label]["f1"]))

    return {
        "method": "midas_record_cluster_percentile_bootstrap",
        "sampling_unit": "midas_record_id",
        "record_count": len(clusters),
        "confidence_level": 0.95,
        "samples": samples,
        "seed": seed,
        "intervals": {
            metric: _percentile_interval(values) for metric, values in distributions.items()
        },
        "per_class_recall_intervals": {
            label: _percentile_interval(values) for label, values in per_class_recall.items()
        },
        "per_class_f1_intervals": {
            label: _percentile_interval(values) for label, values in per_class_f1.items()
        },
    }


def _percentile_interval(values: Sequence[float]) -> dict[str, float]:
    return {
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
    }


def _view_agreement(
    probabilities_6in: np.ndarray,
    probabilities_1ft: np.ndarray,
) -> dict[str, object]:
    six = _validated_probability_matrix(probabilities_6in, expected_rows=len(probabilities_6in))
    one = _validated_probability_matrix(probabilities_1ft, expected_rows=len(probabilities_1ft))
    if len(six) != len(one):
        raise ValueError("15-cm and 30-cm probability rows must align.")
    matches = np.argmax(six, axis=1) == np.argmax(one, axis=1)
    return {
        "lesion_count": len(matches),
        "matching_predicted_label_count": int(matches.sum()),
        "different_predicted_label_count": int((~matches).sum()),
        "predicted_label_agreement": float(matches.mean()),
    }


def _majority_reference(truth: Sequence[str]) -> dict[str, object]:
    support = Counter(truth)
    majority = max(
        PAD_UFES_NATIVE_LABELS,
        key=lambda label: (support[label], -PAD_UFES_NATIVE_LABELS.index(label)),
    )
    predictions = [majority] * len(truth)
    return {
        "description": (
            "Trivial prevalence reference; predicts the most common class for every lesion."
        ),
        "predicted_label": majority,
        "metrics": summarize_metrics(list(truth), predictions, labels=PAD_UFES_NATIVE_LABELS),
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, object] | None:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative.")
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100.")

    data_dir = resolve_project_path(args.data_dir)
    out_dir = resolve_project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = validate_authorized_source_hashes(data_dir)
    release_rows, manifest_rows = load_source_tables(data_dir, validate_files=True)
    prepared = prepare_primary_cohort(release_rows, manifest_rows)
    audit = {**prepared.audit, "source_sha256": source_hashes}
    validate_frozen_audit(audit)
    audit_path = out_dir / "dataset_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote aggregate MRA-MIDAS audit: {audit_path}")
    if args.audit_only:
        return None

    checkpoint_paths = discover_checkpoint_paths(args.checkpoints_dir, folds=args.folds)
    device = _resolve_device(args.device)
    dataset = MraMidasImageDataset(prepared.image_rows)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    truth = prepared.lesion_rows["label"].tolist()
    record_ids = prepared.lesion_rows["record_id"].tolist()

    fold_image_probabilities = []
    fold_reports = []
    for fold_index, checkpoint_path in enumerate(checkpoint_paths):
        print(f"Evaluating fold_{fold_index}: {checkpoint_path}")
        image_probabilities = _predict_checkpoint(
            checkpoint_path,
            loader,
            sample_count=len(prepared.image_rows),
            architecture=args.architecture,
            seed=args.seed,
            device=device,
        )
        lesion_probabilities = aggregate_lesion_probabilities(
            image_probabilities, prepared.image_rows, prepared.lesion_rows
        )
        fold_image_probabilities.append(image_probabilities)
        fold_reports.append(
            {
                "fold_index": fold_index,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "primary_metrics": multiclass_metrics(
                    truth, lesion_probabilities["paired_equal_view_mean"]
                ),
            }
        )

    ensemble_image_probabilities = np.mean(np.stack(fold_image_probabilities, axis=0), axis=0)
    ensemble = aggregate_lesion_probabilities(
        ensemble_image_probabilities, prepared.image_rows, prepared.lesion_rows
    )
    primary_probabilities = ensemble["paired_equal_view_mean"]
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "context": (
            "Experimental external standardized iPhone/iPad research-clinic classification "
            "evidence only; not patient-taken-photo validation, a medical diagnosis, or a "
            "deployment-readiness result."
        ),
        "dataset": audit,
        "model": {
            "candidate": "PAD-UFES-native ConvNeXt-Tiny grouped-CV fold ensemble",
            "architecture": args.architecture,
            "labels": list(PAD_UFES_NATIVE_LABELS),
            "fold_count": args.folds,
            "seed": args.seed,
            "input_mode": "image_only",
            "preprocessing": EXPECTED_PREPROCESSING,
        },
        "protocol": {
            "task": "MRA-MIDAS exact six-class paired standardized clinical-view evaluation",
            "capture_context": MRA_MIDAS_CAPTURE_CONTEXT,
            "workflow_artifact_context": WORKFLOW_ARTIFACT_CONTEXT,
            "cohort": (
                "non_control_pathology_confirmed_exact_mapping_with_resolved_15cm_and_30cm_views"
            ),
            "image_ensemble": "equal_weight_mean_of_five_fold_probability_vectors",
            "repeated_image_aggregation": "mean_within_each_distance",
            "lesion_aggregation": "equal_weight_mean_of_15cm_and_30cm_distance_vectors",
            "uncertainty": "midas_record_id_cluster_percentile_bootstrap",
            "dermoscopy": "excluded",
            "virtual_portal_images": "audit_only_not_scored",
            "demographic_performance": "withheld_due_to_data_quality_and_support",
            "training_calibration_threshold_tuning_or_model_selection_on_mra_midas": False,
        },
        "reference_baselines": {"always_majority_class": _majority_reference(truth)},
        "folds": fold_reports,
        "ensemble": {
            "primary_paired_equal_view_mean": {
                "metrics": multiclass_metrics(truth, primary_probabilities),
                "confidence_intervals": record_cluster_bootstrap_intervals(
                    truth,
                    primary_probabilities,
                    record_ids,
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                ),
            },
            "secondary_same_cohort_views": {
                "15_cm": {"metrics": multiclass_metrics(truth, ensemble["6in"])},
                "30_cm": {"metrics": multiclass_metrics(truth, ensemble["1ft"])},
                "predicted_label_agreement": _view_agreement(ensemble["6in"], ensemble["1ft"]),
            },
        },
        "limitations": [
            "The primary cohort is an exact six-class subset of the released pathology taxonomy; "
            "controls, SCC in situ, and other diagnoses are not scored.",
            "Thirteen metadata filenames cannot be reconciled one-to-one with archive names and "
            "are quarantined rather than guessed.",
            "The standardized clinical photographs were captured on contemporary iPhone/iPad "
            "devices at fixed distances without flash. The public source does not identify the "
            "photographer, and the scored views are not the separately identified portal "
            "submissions, so this is not patient-taken consumer-photo validation.",
            "Only nine resolved virtual images across six exact-class lesion profiles are present, "
            "so patient- or primary-care-provider-submitted portal photographs are not scored.",
            "The released gender values conflict across paired images for 166 lesion profiles, and "
            "skin-tone, race, and ethnicity support is too imbalanced for comparative performance "
            "or broad fairness claims.",
            "Ruler, skin-marker, and other clinical-workflow artifact prevalence has not been "
            "audited or used for filtering.",
            "No MRA-MIDAS images were used for training, calibration, threshold selection, or "
            "model selection in this evaluation.",
        ],
        "artifact_scope": "aggregate_metrics_only_no_per_image_predictions",
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    metrics = report["ensemble"]["primary_paired_equal_view_mean"]["metrics"]
    print(
        "Wrote MRA-MIDAS report: "
        f"{report_path}; accuracy={metrics['accuracy']:.4f} "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    run_evaluation(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
