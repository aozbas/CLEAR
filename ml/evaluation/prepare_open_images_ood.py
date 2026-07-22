"""Prepare a deterministic, attributed obvious-non-skin OOD development cohort.

Pixels, attribution rows, and generated manifests are private ignored artifacts. The public output
is an aggregate summary with source and cohort fingerprints only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from urllib.request import urlopen

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
IMAGES_PER_CATEGORY = 20
MIN_TARGET_BOX_AREA = 0.20
EXPECTED_LICENSE = "https://creativecommons.org/licenses/by/2.0/"
IMAGE_URL_TEMPLATE = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
PROTOCOL = "open_images_obvious_non_skin_v1"
SOURCE_FILES = {
    "class_descriptions": (
        "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
        12_011,
    ),
    "validation_boxes": (
        "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        25_105_048,
    ),
    "validation_metadata": (
        "https://storage.googleapis.com/openimages/2018_04/validation/"
        "validation-images-with-rotation.csv",
        15_245_485,
    ),
    "test_boxes": (
        "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
        77_484_237,
    ),
    "test_metadata": (
        "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
        45_227_339,
    ),
}
EXPECTED_SOURCE_SHA256 = {
    "class_descriptions": "2fa47fb1b87e71c90b9fbee2dc1184eb8c59601bb811ea4424b200653cedff8e",
    "validation_boxes": "d8bbd59410af14835d7733165a7bb8a3f0213981b22dd5077b0b9f7878991ff2",
    "validation_metadata": "ed93a0e121fe345effdfc7359b848dbc64a1ff6778c8c73563157cb500b33a17",
    "test_boxes": "fe22e579b9453875601576859d14ef3304058165de139c66b725e599202bd7b1",
    "test_metadata": "de55c6b8cbda32a79f4a20f28572d54e9c49d527a4006725675764074a51a36d",
}
EXPECTED_COHORT_FINGERPRINT = ""
EXCLUDED_HUMAN_CLASSES = (
    "Person",
    "Man",
    "Woman",
    "Boy",
    "Girl",
    "Human face",
    "Human head",
    "Human body",
    "Human hair",
    "Human arm",
    "Human hand",
    "Human leg",
    "Human foot",
)
CATEGORY_PLAN = {
    "calibration": {
        "vehicles": ("Car", "Bicycle", "Boat", "Train"),
        "animals": ("Cat", "Dog", "Bird", "Horse"),
        "food": ("Apple", "Banana", "Pizza", "Cake"),
        "household": ("Chair", "Table", "Refrigerator", "Television"),
        "outdoor_objects": ("House", "Tree", "Flower", "Traffic sign"),
    },
    "evaluation": {
        "vehicles": ("Airplane", "Bus", "Motorcycle", "Truck"),
        "animals": ("Elephant", "Bear", "Zebra", "Giraffe"),
        "food": ("Orange", "Strawberry", "Sandwich", "Ice cream"),
        "household": ("Bed", "Couch", "Microwave oven", "Washing machine"),
        "outdoor_objects": ("Street light", "Fountain", "Sculpture", "Clock"),
    },
}
MANIFEST_COLUMNS = (
    "partition",
    "semantic_group",
    "class_name",
    "image_id",
    "image_path",
    "license",
    "author",
    "title",
    "sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed Open Images obvious-non-skin OOD cohort."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--images-per-category", type=int, default=IMAGES_PER_CATEGORY)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, *, expected_bytes: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urlopen(url, timeout=60) as response, partial.open("wb") as output:  # noqa: S310
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if expected_bytes is not None and partial.stat().st_size != expected_bytes:
            raise ValueError(
                f"Source size drifted for {url}: expected {expected_bytes}, "
                f"observed {partial.stat().st_size}."
            )
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def download_sources(metadata_dir: Path) -> tuple[dict[str, Path], dict[str, str]]:
    paths = {}
    hashes = {}
    for name, (url, expected_bytes) in SOURCE_FILES.items():
        path = metadata_dir / f"{name}.csv"
        if not path.is_file():
            download_file(url, path, expected_bytes=expected_bytes)
        elif path.stat().st_size != expected_bytes:
            raise ValueError(f"Cached Open Images source size drifted: {path}")
        observed = sha256_file(path)
        expected = EXPECTED_SOURCE_SHA256.get(name)
        if expected is not None and observed != expected:
            raise ValueError(
                f"Open Images source hash drifted for {name}: expected {expected}, "
                f"observed {observed}."
            )
        paths[name] = path
        hashes[name] = observed
    return paths, hashes


def load_class_ids(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    by_name = {name: label_id for label_id, name in rows}
    required = set(EXCLUDED_HUMAN_CLASSES)
    required.update(
        class_name
        for partitions in CATEGORY_PLAN.values()
        for classes in partitions.values()
        for class_name in classes
    )
    missing = sorted(required.difference(by_name))
    if missing:
        raise ValueError(f"Open Images class descriptions are missing: {', '.join(missing)}")
    return {name: by_name[name] for name in sorted(required)}


def _box_area(row: Mapping[str, str]) -> float:
    return (float(row["XMax"]) - float(row["XMin"])) * (float(row["YMax"]) - float(row["YMin"]))


def collect_candidates(
    path: Path,
    *,
    target_ids: Iterable[str],
    excluded_ids: Iterable[str],
) -> dict[str, set[str]]:
    target_set = set(target_ids)
    excluded_set = set(excluded_ids)
    candidates = {label_id: set() for label_id in target_set}
    excluded_images: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = row["ImageID"]
            label_id = row["LabelName"]
            if label_id in excluded_set:
                excluded_images.add(image_id)
            if (
                label_id in target_set
                and row["Confidence"] == "1"
                and row["IsGroupOf"] == "0"
                and row["IsDepiction"] == "0"
                and _box_area(row) >= MIN_TARGET_BOX_AREA
            ):
                candidates[label_id].add(image_id)
    return {
        label_id: image_ids.difference(excluded_images)
        for label_id, image_ids in candidates.items()
    }


def load_metadata(path: Path, candidate_ids: set[str]) -> dict[str, dict[str, str]]:
    metadata = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = row["ImageID"]
            if image_id not in candidate_ids:
                continue
            if row["License"] != EXPECTED_LICENSE:
                continue
            metadata[image_id] = {
                "license": row["License"],
                "author": row["Author"],
                "title": row["Title"],
            }
    return metadata


def stable_candidate_order(
    image_ids: Iterable[str], *, partition: str, class_name: str
) -> list[str]:
    return sorted(
        image_ids,
        key=lambda image_id: hashlib.sha256(
            f"{SEED}:{partition}:{class_name}:{image_id}".encode()
        ).hexdigest(),
    )


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def download_cohort_image(split: str, image_id: str, destination: Path) -> str:
    url = IMAGE_URL_TEMPLATE.format(split=split, image_id=image_id)
    download_file(url, destination)
    try:
        with Image.open(destination) as image:
            if image.format != "JPEG":
                raise ValueError(f"Unexpected Open Images format for {image_id}.")
            if min(image.size) < 64:
                raise ValueError(f"Open Images candidate {image_id} is too small.")
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        destination.unlink(missing_ok=True)
        raise
    return sha256_file(destination)


def cohort_fingerprint(rows: Iterable[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    stable_rows = sorted(
        rows,
        key=lambda row: (
            row["partition"],
            row["semantic_group"],
            row["class_name"],
            row["image_id"],
        ),
    )
    for row in stable_rows:
        values = [
            row["partition"],
            row["semantic_group"],
            row["class_name"],
            row["image_id"],
            row["license"],
            row["sha256"],
        ]
        digest.update(("\t".join(values) + "\n").encode())
    return digest.hexdigest()


def prepare_cohort(out_dir: Path, *, images_per_category: int) -> dict[str, object]:
    if images_per_category != IMAGES_PER_CATEGORY:
        raise ValueError(f"The frozen cohort requires exactly {IMAGES_PER_CATEGORY} images/class.")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_paths, source_hashes = download_sources(out_dir / "metadata")
    class_ids = load_class_ids(source_paths["class_descriptions"])
    excluded_ids = [class_ids[name] for name in EXCLUDED_HUMAN_CLASSES]
    manifest_rows: list[dict[str, str]] = []
    used_image_ids: set[str] = set()

    for partition, groups in CATEGORY_PLAN.items():
        split = "validation" if partition == "calibration" else "test"
        target_names = [name for names in groups.values() for name in names]
        candidates = collect_candidates(
            source_paths[f"{split}_boxes"],
            target_ids=[class_ids[name] for name in target_names],
            excluded_ids=excluded_ids,
        )
        all_candidate_ids = set().union(*candidates.values())
        metadata = load_metadata(source_paths[f"{split}_metadata"], all_candidate_ids)

        for semantic_group, class_names in groups.items():
            for class_name in class_names:
                label_id = class_ids[class_name]
                selected = 0
                for image_id in stable_candidate_order(
                    candidates[label_id], partition=partition, class_name=class_name
                ):
                    if image_id in used_image_ids or image_id not in metadata:
                        continue
                    destination = (
                        out_dir
                        / "images"
                        / partition
                        / safe_slug(semantic_group)
                        / safe_slug(class_name)
                        / f"{image_id}.jpg"
                    )
                    try:
                        image_sha256 = download_cohort_image(split, image_id, destination)
                    except (OSError, ValueError):
                        continue
                    details = metadata[image_id]
                    manifest_rows.append(
                        {
                            "partition": partition,
                            "semantic_group": semantic_group,
                            "class_name": class_name,
                            "image_id": image_id,
                            "image_path": str(destination),
                            "license": details["license"],
                            "author": details["author"],
                            "title": details["title"],
                            "sha256": image_sha256,
                        }
                    )
                    used_image_ids.add(image_id)
                    selected += 1
                    if selected == images_per_category:
                        break
                if selected != images_per_category:
                    raise ValueError(
                        f"Only prepared {selected}/{images_per_category} images for "
                        f"{partition}/{class_name}."
                    )

    manifest_path = out_dir / "attribution_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    fingerprint = cohort_fingerprint(manifest_rows)
    if EXPECTED_COHORT_FINGERPRINT and fingerprint != EXPECTED_COHORT_FINGERPRINT:
        raise ValueError(
            "Open Images cohort fingerprint drifted: "
            f"expected {EXPECTED_COHORT_FINGERPRINT}, observed {fingerprint}."
        )
    expected_total = sum(
        len(classes) * images_per_category
        for groups in CATEGORY_PLAN.values()
        for classes in groups.values()
    )
    if len(manifest_rows) != expected_total:
        raise ValueError(f"Expected {expected_total} OOD images, found {len(manifest_rows)}.")
    summary = {
        "dataset": "Open Images V5",
        "dataset_role": "obvious_non_skin_ood_development",
        "protocol": PROTOCOL,
        "seed": SEED,
        "license_required_per_image": EXPECTED_LICENSE,
        "images_per_category": images_per_category,
        "minimum_target_box_area": MIN_TARGET_BOX_AREA,
        "excluded_human_classes": list(EXCLUDED_HUMAN_CLASSES),
        "category_plan": CATEGORY_PLAN,
        "counts_by_partition": {
            partition: sum(row["partition"] == partition for row in manifest_rows)
            for partition in CATEGORY_PLAN
        },
        "source_sha256": source_hashes,
        "manifest_sha256": sha256_file(manifest_path),
        "cohort_fingerprint": fingerprint,
        "privacy": {
            "pixels_tracked": False,
            "attribution_manifest_tracked": False,
            "public_output_contains_image_identifiers": False,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Prepared {len(manifest_rows)} Open Images OOD images; fingerprint={fingerprint}")
    return summary


def main() -> None:
    args = parse_args()
    prepare_cohort(args.out_dir, images_per_category=args.images_per_category)


if __name__ == "__main__":
    main()
