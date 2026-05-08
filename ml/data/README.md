# ml/data/

Datasets and label conventions for the skin lesion classifier.

This folder is **gitignored** for image content — only this README and any small mapping files belong in version control. Raw images live locally (or on cloud storage) and are downloaded via a script.

## Layout (local, not committed)
```
ml/data/
├── README.md           # this file
├── raw/                # untouched downloads from the dataset source
│   └── <dataset-name>/
├── processed/          # resized / normalized images ready for training
│   └── <dataset-name>/
└── splits/             # train/val/test CSVs
    └── <dataset-name>.csv
```

## Canonical label set

All datasets must be translated to this canonical label set before training. The model only ever sees these strings.

**Current canonical labels (HAM10000-supported, Phases 1–2):**

| canonical label         | short code | description                                   |
|-------------------------|------------|-----------------------------------------------|
| `melanoma`              | `mel`      | malignant melanoma                            |
| `nevus`                 | `nv`       | melanocytic nevus (common mole)               |
| `basal_cell_carcinoma`  | `bcc`      | basal cell carcinoma                          |
| `actinic_keratosis`     | `akiec`    | actinic keratosis / intraepithelial carcinoma |
| `benign_keratosis`      | `bkl`      | benign keratosis-like lesions                 |
| `dermatofibroma`        | `df`       | dermatofibroma                                |
| `vascular_lesion`       | `vasc`     | vascular lesions (angiomas, hemorrhage)       |

**Future canonical labels (added in Phase 3 when multi-dataset training begins):**

| canonical label           | short code | available in         |
|---------------------------|------------|----------------------|
| `squamous_cell_carcinoma` | `scc`      | ISIC Archive         |
| `seborrheic_keratosis`    | `sk`       | ISIC Archive         |

> HAM10000 folds SCC cases into `actinic_keratosis` and has no separate `seborrheic_keratosis` class.
> These two labels must not appear in training data until a dataset that supports them is added.

Use `snake_case` canonical names in code and DB rows. Use the short code only when matching dataset filenames.

## Dataset → canonical translation

### HAM10000 (`dx` column)
| HAM10000 | canonical                |
|----------|--------------------------|
| `mel`    | `melanoma`               |
| `nv`     | `nevus`                  |
| `bcc`    | `basal_cell_carcinoma`   |
| `akiec`  | `actinic_keratosis`      |
| `bkl`    | `benign_keratosis`       |
| `df`     | `dermatofibroma`         |
| `vasc`   | `vascular_lesion`        |

HAM10000 does not have a separate `squamous_cell_carcinoma` class — SCC cases are folded into `akiec`. If a dataset distinguishes them, keep them separate.

### ISIC Archive (`diagnosis` field)
| ISIC                          | canonical                 |
|-------------------------------|---------------------------|
| `melanoma`                    | `melanoma`                |
| `nevus`                       | `nevus`                   |
| `basal cell carcinoma`        | `basal_cell_carcinoma`    |
| `squamous cell carcinoma`     | `squamous_cell_carcinoma` |
| `actinic keratosis`           | `actinic_keratosis`       |
| `seborrheic keratosis`        | `seborrheic_keratosis`    |
| `solar lentigo`               | `benign_keratosis`        |
| `lichenoid keratosis`         | `benign_keratosis`        |
| `dermatofibroma`              | `dermatofibroma`          |
| `vascular lesion`             | `vascular_lesion`         |
| `angioma` / `angiokeratoma`   | `vascular_lesion`         |

ISIC uses lowercase free-text diagnoses; normalize whitespace and case before mapping.

## Rules
1. **Never train on un-normalized labels.** A label that doesn't appear in the canonical table must either be added here (with justification in [../../docs/decisions.md](../../docs/decisions.md)) or excluded from the training set.
2. **One canonical name per concept.** Don't introduce synonyms (`mole` and `nevus`, etc.).
3. **Snake_case only** in code, DB, and API responses. Display strings (e.g. "Basal Cell Carcinoma") are a UI concern and live in the mobile app.
4. **Class imbalance is real.** HAM10000 is ~67% nevus. Document any reweighting / sampling strategy in `docs/decisions.md`.
5. **Binary fallback.** For the Phase-1 binary baseline, group the 7 HAM10000 labels as:
   - `suspicious` = `melanoma`, `basal_cell_carcinoma`, `actinic_keratosis`
   - `non_suspicious` = `nevus`, `benign_keratosis`, `dermatofibroma`, `vascular_lesion`

## Adding a new dataset
1. Drop raw files in `raw/<dataset-name>/`.
2. Add a translation table in this README under "Dataset → canonical translation".
3. Write a script in `ml/training/` that emits `splits/<dataset-name>.csv` with columns: `image_path,label` (label = canonical).
4. Note the choice (and any excluded classes) in [../../docs/decisions.md](../../docs/decisions.md).
