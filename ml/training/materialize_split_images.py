"""Materialize lossless 224x224 image inputs for repeatable cloud training."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from ml.training.prepare_pad_ufes import project_relative
from ml.training.train import resolve_project_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "pad_ufes_224"
IMAGE_SIZE = (224, 224)
REQUIRED_COLUMNS = {"split", "image_path", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize lossless 224x224 images and rewrite a split CSV."
    )
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def materialize_split(
    split_csv: Path,
    out_csv: Path,
    image_dir: Path,
    *,
    workers: int = 8,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if workers <= 0:
        raise ValueError("workers must be positive.")

    split_csv = resolve_project_path(Path(split_csv))
    out_csv = resolve_project_path(Path(out_csv))
    image_dir = resolve_project_path(Path(image_dir))
    if split_csv.resolve() == out_csv.resolve():
        raise ValueError("out_csv must differ from split_csv.")

    rows = pd.read_csv(split_csv)
    missing_columns = REQUIRED_COLUMNS.difference(rows.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Split CSV is missing required columns: {missing}")

    source_summary_path = split_csv.with_suffix(".summary.json")
    if not source_summary_path.exists():
        raise FileNotFoundError(f"Missing source split summary: {source_summary_path}")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("image_count") != len(rows):
        raise ValueError("Source split summary image_count does not match the split CSV.")

    image_dir.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(set(str(path) for path in rows["image_path"]))

    def process(source_path: str) -> tuple[str, str, bool]:
        return _materialize_one(
            source_path,
            image_dir=image_dir,
            overwrite=overwrite,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process, source_paths))

    path_mapping = {source: target for source, target, _created in results}
    output = rows.copy()
    output["image_path"] = output["image_path"].astype(str).map(path_mapping)
    if bool(output["image_path"].isna().any()):
        raise ValueError("A split image path was not materialized.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_csv, index=False)
    created_count = sum(created for _source, _target, created in results)
    materialization = {
        "format": "PNG",
        "lossless": True,
        "size": list(IMAGE_SIZE),
        "interpolation": "bilinear",
        "antialias": True,
        "unique_image_count": len(source_paths),
        "created_image_count": created_count,
        "reused_image_count": len(source_paths) - created_count,
    }
    output_summary = {
        **source_summary,
        "materialization": materialization,
    }
    out_csv.with_suffix(".summary.json").write_text(
        json.dumps(output_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output, materialization


def _materialize_one(
    source_path: str,
    *,
    image_dir: Path,
    overwrite: bool,
) -> tuple[str, str, bool]:
    source = resolve_project_path(Path(source_path))
    if not source.exists():
        raise FileNotFoundError(f"Missing source image: {source}")

    digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()
    target = image_dir / f"{digest}.png"
    if target.exists() and not overwrite:
        _validate_materialized_image(target)
        return source_path, project_relative(target), False

    with Image.open(source) as opened:
        image = opened.convert("RGB")
    resized = resize(
        image,
        list(IMAGE_SIZE),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    temporary = target.with_suffix(".tmp")
    try:
        resized.save(temporary, format="PNG", compress_level=1)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    _validate_materialized_image(target)
    return source_path, project_relative(target), True


def _validate_materialized_image(path: Path) -> None:
    with Image.open(path) as image:
        if image.size != IMAGE_SIZE or image.mode != "RGB":
            raise ValueError(
                f"Materialized image has unexpected shape or mode: {path} "
                f"size={image.size} mode={image.mode}"
            )


def main() -> None:
    args = parse_args()
    output, summary = materialize_split(
        args.split_csv,
        args.out_csv,
        args.image_dir,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote {len(output):,} rows to {project_relative(resolve_project_path(args.out_csv))}; "
        f"created={summary['created_image_count']} reused={summary['reused_image_count']}"
    )


if __name__ == "__main__":
    main()
