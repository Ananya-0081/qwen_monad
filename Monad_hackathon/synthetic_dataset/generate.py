"""Create a conservative, reproducible black-box synthetic currency dataset.

The script reads only the two configured source roots and writes only below
the synthetic_dataset directory. It does not create train/validation/test
splits and never edits source files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    import cv2
except ImportError:  # pragma: no cover - checked at runtime by main
    cv2 = None


CLASSES = ("10", "20", "50", "100", "200", "500")
RAW_CLASS_MAP = {
    "ten_new": "10",
    "ten_old": "10",
    "twenty_new": "20",
    "twenty_old": "20",
    "fifty_new": "50",
    "fifty_old": "50",
    "hundred_new": "100",
    "hundred_old": "100",
    "two_hundred": "200",
    "five_hundred": "500",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
OUTPUT_SIZE = (640, 480)


@dataclass
class Source:
    source_id: str
    path: Path
    denomination: str
    source_type: str
    source_group: str
    physical_note_id: str = "UNKNOWN"
    quality_status: str = "GOOD"
    duplicate_group: str = ""
    selected_for_synthesis: str = "yes"
    sha256: str = ""
    width: int = 0
    height: int = 0
    file_size: int = 0
    hash_canonical: bool = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def source_candidates(raw_root: Path, real_root: Path) -> list[tuple[Path, str, str, str]]:
    candidates: list[tuple[Path, str, str, str]] = []
    for folder_name, denomination in RAW_CLASS_MAP.items():
        folder = raw_root / folder_name
        if folder.is_dir():
            candidates.extend((path, denomination, "kaggle_or_raw", folder_name)
                              for path in sorted(folder.iterdir())
                              if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    for denomination in CLASSES:
        folder = real_root / denomination
        if folder.is_dir():
            candidates.extend((path, denomination, "real_camera", f"real_currency/{denomination}")
                              for path in sorted(folder.iterdir())
                              if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    return candidates


def read_real_metadata(real_root: Path) -> dict[str, dict[str, str]]:
    metadata_path = real_root / "manifest.csv"
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        return {row.get("image_path", ""): row for row in rows if row.get("image_path")}


def assess_quality(path: Path, source_type: str, metadata: dict[str, str]) -> tuple[str, str]:
    """Return a conservative status and reason without altering the source."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            array = np.asarray(image.convert("L"), dtype=np.float32)
    except Exception as error:
        return "EXCLUDE", f"unreadable_or_corrupt: {error}"

    if width < 160 or height < 160:
        return "EXCLUDE", "extremely_low_resolution"
    mean = float(array.mean())
    deviation = float(array.std())
    if deviation < 4:
        return "EXCLUDE", "blank_or_nearly_blank"
    if mean < 8 or mean > 249:
        return "REVIEW", "extreme_brightness_or_darkness"

    if source_type == "real_camera":
        include = metadata.get("include_for_training", "").strip().lower()
        duplicate = metadata.get("duplicate", "").strip().upper()
        image_quality = metadata.get("image_quality", "").strip().lower()
        if duplicate == "YES" or include == "no":
            return "EXCLUDE", "marked_duplicate_or_not_for_training"
        if image_quality in {"bad", "unacceptable", "corrupt"}:
            return "EXCLUDE", f"manifest_quality={image_quality}"
        if image_quality in {"acceptable", "review"} or include == "review":
            return "REVIEW", "manifest_requires_review"
    return "GOOD", ""


def build_sources(raw_root: Path, real_root: Path) -> tuple[list[Source], list[dict[str, str]], dict[str, list[Source]]]:
    metadata = read_real_metadata(real_root)
    sources: list[Source] = []
    label_reviews: list[dict[str, str]] = []
    by_hash: dict[str, list[Source]] = defaultdict(list)
    for index, (path, denomination, source_type, source_group) in enumerate(
        source_candidates(raw_root, real_root), start=1
    ):
        row = metadata.get(f"{denomination}/{path.name}", {})
        quality, _ = assess_quality(path, source_type, row)
        source = Source(
            source_id=f"SRC_{index:06d}",
            path=path,
            denomination=denomination,
            source_type=source_type,
            source_group=source_group,
            physical_note_id=row.get("physical_note_id", "UNKNOWN") or "UNKNOWN",
            quality_status=quality,
            selected_for_synthesis="yes" if quality == "GOOD" else "no",
            file_size=path.stat().st_size,
        )
        try:
            with Image.open(path) as image:
                source.width, source.height = image.size
        except Exception:
            pass
        source.sha256 = sha256_file(path)
        by_hash[source.sha256].append(source)
        sources.append(source)
        if denomination == "100" and "watermark" in path.name.lower():
            source.quality_status = "REVIEW"
            source.selected_for_synthesis = "no"
            label_reviews.append({
                "image_path": path.resolve().relative_to(Path.cwd().resolve()).as_posix(),
                "folder_label": denomination,
                "suspected_label": "UNKNOWN",
                "reason": "filename suggests possible watermark; inspect visually",
                "action_required": "human verification before synthesis",
            })

    duplicate_groups: dict[str, list[Source]] = {}
    for digest, group in by_hash.items():
        if len(group) > 1:
            duplicate_id = f"DUP_{len(duplicate_groups) + 1:04d}"
            duplicate_groups[duplicate_id] = group
            for duplicate in group:
                duplicate.duplicate_group = duplicate_id
            for duplicate in group[1:]:
                duplicate.hash_canonical = False
                duplicate.quality_status = "EXCLUDE"
                duplicate.selected_for_synthesis = "no"
    unique_by_class: dict[str, list[Source]] = defaultdict(list)
    for source in sources:
        if source.hash_canonical:
            unique_by_class[source.denomination].append(source)
    return sources, label_reviews, unique_by_class


def make_manifest_row(source: Source) -> dict[str, Any]:
    relative = source.path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    return {
        "source_id": source.source_id,
        "original_path": relative,
        "source_group": source.source_group,
        "denomination": source.denomination,
        "file_name": source.path.name,
        "file_extension": source.path.suffix.lower(),
        "width": source.width,
        "height": source.height,
        "aspect_ratio": round(source.width / source.height, 6) if source.height else "",
        "file_size": source.file_size,
        "source_type": source.source_type,
        "duplicate_group": source.duplicate_group,
        "quality_status": source.quality_status,
        "selected_for_synthesis": source.selected_for_synthesis,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def grabcut_foreground(image: Image.Image) -> tuple[Image.Image | None, str]:
    if cv2 is None:
        return None, "opencv_unavailable"
    working_image = image.convert("RGB")
    working_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    rgb = np.asarray(working_image)
    height, width = rgb.shape[:2]
    if width < 32 or height < 32:
        return None, "image_too_small"
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((height, width), np.uint8)
    margin_x = max(2, int(width * 0.04))
    margin_y = max(2, int(height * 0.04))
    rectangle = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rectangle, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error as error:
        return None, f"grabcut_failed: {error}"
    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    area = int(np.count_nonzero(foreground))
    fraction = area / float(width * height)
    if fraction < 0.08 or fraction > 0.98:
        return None, f"segmentation_review foreground_fraction={fraction:.3f}"
    alpha = Image.fromarray(foreground, mode="L")
    rgba = working_image.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba, ""


def matte_background(size: tuple[int, int], rng: random.Random) -> Image.Image:
    width, height = size
    y, x = np.mgrid[0:height, 0:width]
    center_x = width * rng.uniform(0.42, 0.58)
    center_y = height * rng.uniform(0.42, 0.58)
    distance = np.sqrt(((x - center_x) / width) ** 2 + ((y - center_y) / height) ** 2)
    base = rng.uniform(7, 15)
    gradient = np.clip(1.0 - distance * rng.uniform(0.12, 0.28), 0.75, 1.05)
    values = np.clip(base * gradient, 5, 25).astype(np.uint8)
    rgb = np.dstack([values, values, values])
    return Image.fromarray(rgb, mode="RGB")


def paste_variant(foreground: Image.Image, rng: random.Random, variant_number: int) -> tuple[Image.Image, dict[str, Any]]:
    canvas = matte_background(OUTPUT_SIZE, rng).convert("RGBA")
    note = foreground.copy()
    rotation = rng.choice([0, -5, 5, -10, 10, -15, 15, -20, 20, -30, 30, -45, 45, 90, 180])
    scale_name, scale = rng.choice([("small", 0.38), ("medium", 0.52), ("large", 0.67), ("close", 0.80)])
    note.thumbnail((int(OUTPUT_SIZE[0] * scale), int(OUTPUT_SIZE[1] * scale)), Image.Resampling.LANCZOS)
    note = note.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
    perspective_strength = rng.uniform(-0.06, 0.06)
    x_fraction = rng.uniform(0.22, 0.78)
    y_fraction = rng.uniform(0.22, 0.78)
    x = int(OUTPUT_SIZE[0] * x_fraction - note.width / 2)
    y = int(OUTPUT_SIZE[1] * y_fraction - note.height / 2)
    x = max(-int(note.width * 0.15), min(OUTPUT_SIZE[0] - int(note.width * 0.85), x))
    y = max(-int(note.height * 0.15), min(OUTPUT_SIZE[1] - int(note.height * 0.85), y))

    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_note = note.getchannel("A").filter(ImageFilter.GaussianBlur(radius=rng.uniform(4, 10)))
    shadow_color = Image.new("RGBA", note.size, (0, 0, 0, rng.randint(35, 75)))
    shadow_color.putalpha(shadow_note.point(lambda value: int(value * 0.45)))
    shadow.alpha_composite(shadow_color, (x + rng.randint(3, 8), y + rng.randint(4, 10)))
    if rng.random() < 0.72:
        canvas.alpha_composite(shadow)
        shadow_mode = "soft"
    else:
        shadow_mode = "none"

    canvas.alpha_composite(note, (x, y))
    rgb = canvas.convert("RGB")
    brightness_level, brightness = rng.choice([
        ("LOW", 0.88), ("MEDIUM-LOW", 0.95), ("MEDIUM", 1.0),
        ("MEDIUM-HIGH", 1.06), ("HIGH", 1.13),
    ])
    contrast = rng.choice([0.94, 1.0, 1.06])
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    color_temperature = rng.choice(["neutral", "warm", "cool"])
    if color_temperature != "neutral":
        channels = np.asarray(rgb).astype(np.int16)
        if color_temperature == "warm":
            channels[:, :, 0] += 3
            channels[:, :, 2] -= 3
        else:
            channels[:, :, 0] -= 3
            channels[:, :, 2] += 3
        rgb = Image.fromarray(np.clip(channels, 0, 255).astype(np.uint8))

    glare = "none"
    if rng.random() < 0.12:
        overlay = Image.new("RGBA", rgb.size, (255, 255, 255, 0))
        band = Image.new("L", (rgb.width, rgb.height), 0)
        band_draw = np.zeros((rgb.height, rgb.width), dtype=np.uint8)
        band_draw[:, int(rgb.width * rng.uniform(0.35, 0.7)):int(rgb.width * rng.uniform(0.7, 0.9))] = rng.randint(8, 22)
        overlay.putalpha(Image.fromarray(band_draw).filter(ImageFilter.GaussianBlur(24)))
        rgb = Image.alpha_composite(rgb.convert("RGBA"), overlay).convert("RGB")
        glare = "mild"

    blur = "none"
    if rng.random() < 0.16:
        radius = rng.choice([0.35, 0.6, 0.9])
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=radius))
        blur = f"gaussian_{radius}"
    noise = "none"
    if rng.random() < 0.25:
        array = np.asarray(rgb).astype(np.int16)
        noise_array = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, 1.4, array.shape)
        rgb = Image.fromarray(np.clip(array + noise_array, 0, 255).astype(np.uint8))
        noise = "mild_gaussian"
    vignette = rng.choice(["none", "mild", "moderate"])
    if vignette != "none":
        array = np.asarray(rgb).astype(np.float32)
        height, width = array.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        distance = np.sqrt(((xx - width / 2) / width) ** 2 + ((yy - height / 2) / height) ** 2)
        strength = 0.10 if vignette == "mild" else 0.17
        factor = 1.0 - np.clip(distance * strength, 0, 0.2)
        rgb = Image.fromarray(np.clip(array * factor[..., None], 0, 255).astype(np.uint8))

    return rgb, {
        "variant_id": f"V{variant_number:03d}",
        "rotation": rotation,
        "scale": scale_name,
        "translation_x": round(x / OUTPUT_SIZE[0], 5),
        "translation_y": round(y / OUTPUT_SIZE[1], 5),
        "perspective_strength": round(perspective_strength, 5),
        "brightness": brightness_level,
        "contrast": contrast,
        "color_temperature": color_temperature,
        "shadow": shadow_mode,
        "glare": glare,
        "blur": blur,
        "noise": noise,
        "vignette": vignette,
    }


def choose_variant_count(denomination: str, source_count: int) -> int:
    if denomination == "100":
        return 8
    if source_count < 40:
        return 12
    return 10


def generate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd()
    output = root / "synthetic_dataset"
    raw_root = root / args.raw_dir
    real_root = root / args.real_dir
    if output.exists() and any(output.iterdir()) and not args.allow_existing_output:
        raise FileExistsError(
            f"{output} already contains files. Use --allow-existing-output only to replace generated output."
        )
    if args.allow_existing_output and output.exists():
        for directory in [output / "black_box", output / "masks", output / "metadata", output / "previews"]:
            if directory.exists():
                shutil.rmtree(directory)
    for directory in [output / "black_box", output / "masks", output / "metadata", output / "previews"]:
        directory.mkdir(parents=True, exist_ok=True)
    for denomination in CLASSES:
        (output / "black_box" / denomination).mkdir(parents=True, exist_ok=True)

    candidates = source_candidates(raw_root, real_root)
    if not candidates:
        raise RuntimeError("No supported source images were found in the configured input paths.")
    sources, label_reviews, unique_by_class = build_sources(raw_root, real_root)
    manifest_fields = [
        "source_id", "original_path", "source_group", "denomination", "file_name",
        "file_extension", "width", "height", "aspect_ratio", "file_size", "source_type",
        "duplicate_group", "quality_status", "selected_for_synthesis",
    ]
    write_csv(output / "metadata" / "source_manifest.csv", [make_manifest_row(source) for source in sources], manifest_fields)
    write_csv(output / "metadata" / "label_review.csv", label_reviews,
              ["image_path", "folder_label", "suspected_label", "reason", "action_required"])

    rng = random.Random(SEED)
    synthetic_rows: list[dict[str, Any]] = []
    counts = Counter()
    segmentation_failures = 0
    quality_exclusions = 0
    review_count = sum(source.quality_status == "REVIEW" for source in sources)
    usable_sources: dict[str, list[Source]] = defaultdict(list)
    for denomination in CLASSES:
        for source in unique_by_class.get(denomination, []):
            if source.quality_status == "GOOD":
                usable_sources[denomination].append(source)
            else:
                quality_exclusions += 1

    for denomination in CLASSES:
        source_list = usable_sources[denomination]
        variants_per_source = choose_variant_count(denomination, len(source_list))
        for source in source_list:
            try:
                image = Image.open(source.path).convert("RGB")
            except Exception:
                segmentation_failures += 1
                continue
            foreground, reason = grabcut_foreground(image)
            if foreground is None:
                segmentation_failures += 1
                source.quality_status = "SEGMENTATION_REVIEW"
                source.selected_for_synthesis = "no"
                continue
            mask_path = output / "masks" / f"{source.source_id}.png"
            foreground.getchannel("A").save(mask_path)
            preview = [image.resize((160, 120)), foreground.convert("RGB").resize((160, 120))]
            for variant_number in range(1, variants_per_source + 1):
                result, parameters = paste_variant(foreground, rng, variant_number)
                synthetic_id = f"SYN_{denomination}_{len(synthetic_rows) + 1:07d}"
                output_path = output / "black_box" / denomination / f"{synthetic_id}.jpg"
                result.save(output_path, quality=91, optimize=True)
                if variant_number <= 5:
                    preview.append(result.resize((160, 120)))
                synthetic_rows.append({
                    "synthetic_id": synthetic_id,
                    "source_id": source.source_id,
                    "source_path": source.path.resolve().relative_to(root.resolve()).as_posix(),
                    "source_type": source.source_type,
                    "denomination": denomination,
                    "physical_note_id": source.physical_note_id,
                    **parameters,
                    "image_width": OUTPUT_SIZE[0],
                    "image_height": OUTPUT_SIZE[1],
                    "image_type": "synthetic",
                })
                counts[denomination] += 1
            preview_sheet = Image.new("RGB", (160 * len(preview), 120), "black")
            for position, tile in enumerate(preview):
                preview_sheet.paste(tile.convert("RGB"), (position * 160, 0))
            preview_sheet.save(output / "previews" / f"{source.source_id}.jpg", quality=88)

    synthetic_fields = [
        "synthetic_id", "source_id", "source_path", "source_type", "denomination",
        "physical_note_id", "variant_id", "rotation", "scale", "translation_x",
        "translation_y", "perspective_strength", "brightness", "contrast",
        "color_temperature", "shadow", "glare", "blur", "noise", "vignette",
        "image_width", "image_height", "image_type",
    ]
    write_csv(output / "metadata" / "synthetic_manifest.csv", synthetic_rows, synthetic_fields)
    duplicate_groups = len({source.duplicate_group for source in sources if source.duplicate_group})
    report = {
        "seed": SEED,
        "input_paths": {"raw": raw_root.as_posix(), "real_camera": real_root.as_posix()},
        "output_path": output.as_posix(),
        "classes": list(CLASSES),
        "excluded_raw_class": "two_thousand",
        "total_source_images": len(sources),
        "unique_source_images": sum(len(group) for group in unique_by_class.values()),
        "exact_duplicate_groups": duplicate_groups,
        "exact_duplicate_files_excluded": sum(not source.hash_canonical for source in sources),
        "quality_exclusions": quality_exclusions,
        "review_required": review_count,
        "label_review_cases": len(label_reviews),
        "segmentation_failures": segmentation_failures,
        "synthetic_total": len(synthetic_rows),
        "per_denomination": {
            denomination: {
                "source_count": sum(source.denomination == denomination for source in sources),
                "usable_source_count": len(usable_sources[denomination]),
                "synthetic_count": counts[denomination],
                "final_available_count": counts[denomination],
            }
            for denomination in CLASSES
        },
        "warnings": [
            "Synthetic images are not independent physical notes.",
            "Group source images, physical notes, bursts, and synthetic families before future splitting.",
            "GrabCut masks are conservative candidates and require visual review.",
            "No train/val/test directories were created.",
        ],
    }
    (output / "metadata" / "generation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data2/data/raw")
    parser.add_argument("--real-dir", default="data2/data/real_currency")
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()
    if cv2 is None:
        print("ERROR: OpenCV is required for the offline segmentation fallback.", file=sys.stderr)
        return 2
    try:
        report = generate(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("=" * 50)
    print("SYNTHETIC DATASET GENERATION COMPLETE")
    print("=" * 50)
    print(f"Source images: {report['total_source_images']}")
    print(f"Unique source images: {report['unique_source_images']}")
    print(f"Excluded: {report['quality_exclusions']}")
    print(f"Duplicates: {report['exact_duplicate_files_excluded']} files in {report['exact_duplicate_groups']} groups")
    print(f"Review required: {report['review_required']}")
    for denomination in CLASSES:
        item = report["per_denomination"][denomination]
        print(f"₹{denomination}: Source: {item['usable_source_count']} | Synthetic: {item['synthetic_count']}")
    print(f"Total synthetic images: {report['synthetic_total']}")
    print(f"Segmentation failures: {report['segmentation_failures']}")
    print(f"Quality exclusions: {report['quality_exclusions']}")
    print(f"Label-review cases: {report['label_review_cases']}")
    print("Output: synthetic_dataset/")
    print("WARNING: inspect masks/previews before using the synthetic images.")
    print("WARNING: no train/val/test splits were created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())