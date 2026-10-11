# Synthetic Black-Box Dataset

This directory is generated only from:

- `data2/data/raw/`
- `data2/data/real_currency/`

The generator excludes the raw `two_thousand` class and creates only `10`,
`20`, `50`, `100`, `200`, and `500`. Existing project data is never edited.

## What is generated

- `black_box/<denomination>/`: synthetic 640x480 matte-black-box images.
- `masks/`: alpha masks produced by the offline segmentation step.
- `previews/`: compact source, mask, and variant contact sheets.
- `metadata/source_manifest.csv`: every source image and its status.
- `metadata/synthetic_manifest.csv`: every generated image and transform values.
- `metadata/label_review.csv`: suspicious items requiring human review.
- `metadata/generation_report.json`: reproducible counts and warnings.

No `train/`, `val/`, or `test/` directories are created. Source images,
duplicates, and synthetic families must be grouped before a future split.

## Processing

Exact SHA-256 duplicates are retained in the source manifest but only one
canonical copy can generate variants. Source files marked `REVIEW`, `EXCLUDE`,
or segmentation failures are not synthesized. The generator uses an offline
OpenCV GrabCut fallback because `rembg` is optional; masks and previews must be
visually inspected before use.

Variants use a fixed seed (`42`) and conservative black-box transformations:
position, rotation, scale, mild perspective metadata, LED brightness, contrast,
warm/cool color temperature, soft shadow, rare glare, mild blur, noise, and
vignette. Synthetic data increases environmental variety; it does not create
new physical notes.

## Reproduce

From the project root, with the project virtual environment active:

```powershell
python synthetic_dataset/generate.py
```

To regenerate after reviewing this output directory, explicitly opt in:

```powershell
python synthetic_dataset/generate.py --allow-existing-output
```

The generator does not train a model, modify source files, or create dataset
splits.