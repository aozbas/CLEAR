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

## HAM10000 dataset notes

**Source:** Tschandl et al. 2018, *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions* (Sci. Data 5:180161). 10,015 dermatoscopy images of 7,470 unique lesions. Distributed on Kaggle as `kmader/skin-cancer-mnist-ham10000`.

### Metadata schema (`HAM10000_metadata.csv`)
| column | meaning |
|--------|---------|
| `lesion_id` | physical lesion identifier; one or more images may share a `lesion_id` (different angles or visits of the same spot) |
| `image_id` | unique image identifier; matches the `.jpg` filename |
| `dx` | short diagnosis code (see canonical translation above) |
| `dx_type` | how the diagnosis was confirmed (see below) |
| `age` | patient age in years (nullable) |
| `sex` | patient sex |
| `localization` | body site of the lesion |

Images are split across two sibling folders: `HAM10000_images_part_1/` (~5,000 images) and `HAM10000_images_part_2/` (~5,015 images). Filenames match `image_id`.

### `dx_type` values
| code | meaning | reliability |
|------|---------|-------------|
| `histo` | Histopathology — biopsy + lab analysis | Highest; ground truth |
| `follow_up` | Lesion observed clinically over time and didn't change → assumed benign | Medium; only valid for benign cases |
| `consensus` | Multiple experts agreed from clinical/dermoscopic images alone | Medium |
| `confocal` | In-vivo confocal microscopy (non-invasive imaging) | Medium-high |

`dx_type` describes how the *diagnosis* was confirmed, **not** how the *image* was captured. All HAM10000 images are dermatoscopy photos regardless of `dx_type`.

### Multiple images per lesion
About 18% of lesions (~1,200 out of 7,470) have more than one image — typically 2-3, occasionally up to 6. This drives the design of `ml/training/prepare_ham10000.py`, which splits at the lesion level so the same physical spot never appears in both train and test (see [`../../docs/decisions.md`](../../docs/decisions.md)).

### Known data quirks
- **Inconsistent localization labels (2 lesions).** `HAM_0000871` and `HAM_0001726` each have one row labeled `trunk` while the other rows of the same lesion are labeled `chest` or `back`. Most likely cause: different labelers used different granularities (the generic "trunk" vs the specific "chest"/"back"). This does not affect Phase 1 since `localization` is not in the prepare script output, but would need normalization (e.g., majority vote per lesion) if we ever stratify by body site.
- **Dermatoscope vignette artifact.** Many images include a circular dark border from the dermatoscope hardware (a contact-lens-like attachment with built-in lighting). This is unrelated to `dx_type` — it appears across all four values. The model may learn this border as a shortcut feature; worth checking with a saliency map after training.
- **Measurement ruler markings.** A subset of images has a millimeter ruler with tick marks visible along one edge of the frame (dermatologists place a ruler next to the lesion to track size over time). Examples: `ISIC_0025964`, `ISIC_0024367`. **This is a known shortcut-learning hazard:** dermatologists are more likely to use a ruler on lesions they already suspect are malignant, so in the training data the ruler is correlated with `mel`. Models trained on HAM10000 have repeatedly been shown to latch onto this artifact instead of actual lesion features (see Bissoto et al. 2020, Winkler et al. 2019). For Phase 1 this is acknowledged but not actively mitigated; if test accuracy looks too good, run a saliency map on melanoma predictions to check whether the model is attending to the ruler region rather than the lesion itself.

### Class imbalance
~67% of images are `nv` (nevus). Rare classes are `df` (115 images) and `vasc` (142 images). Any reweighting or sampling strategy is recorded in [`../../docs/decisions.md`](../../docs/decisions.md).

### Exploration
For visual exploration of the above (class distribution, lesion structure, image samples by class and `dx_type`, color/brightness statistics), see [`../notebooks/01_explore_ham10000.ipynb`](../notebooks/01_explore_ham10000.ipynb).

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
