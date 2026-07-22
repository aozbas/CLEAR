# CLEAR Experimental Demo Model Evidence Card

## Intended scope

CLEAR is an educational and technical demonstration of image-classification plumbing. It returns a
single experimental category for a submitted skin-lesion image. It is not a medical device,
diagnostic system, triage tool, screening tool, clinical decision aid, or consumer reassurance tool.

## Model actually wired into the demo

Static backend and inference configuration currently selects:

- Architecture: ResNet18
- Training domain: HAM10000 dermoscopy images
- Output vocabulary: melanoma, nevus, basal cell carcinoma, actinic keratosis, benign keratosis,
  dermatofibroma, and vascular lesion
- Configured model-version string: `ham10000-resnet18-baseline-2026-05-18`
- Display value: the top softmax output, labeled in the UI as an **uncalibrated model score**

The repository does not track the model weights. This wiring was determined by static code and
configuration inspection; this documentation does not load or evaluate the model.

## Training-data and checkpoint rights boundary

The official [HAM10000 dataset record](https://doi.org/10.7910/DVN/DBW86T) designates the dataset
under the Creative Commons Attribution-NonCommercial 4.0 International terms. The accompanying
[dataset paper](https://doi.org/10.1038/sdata.2018.161) should be cited when this training source is
described.

Those dataset terms are separate from any license selected for CLEAR's source code. A code license
does not relicense HAM10000, any other dataset, or a trained checkpoint. Before publicly hosting or
distributing the configured checkpoint, the project owner must document its exact provenance,
confirm that the intended use complies with all source terms, retain required attribution, and keep
the use noncommercial unless separate permission supports a broader use.

## Central limitation

HAM10000 is a dermoscopy-domain dataset. The public demo accepts ordinary JPEG or PNG images, but
there is no evidence that the wired model generalizes to patient- or consumer-taken smartphone
photos. Its score must not be interpreted as disease probability, severity, urgency, or certainty.

## Separate research evidence—not demo validation

Research on PAD-UFES phone-photo data found ConvNeXt-Tiny to be the strongest internal candidate
(five-fold macro F1 0.6555; pooled macro F1 0.6609). SCC remained weak (F1 0.2444), and PAD-UFES had
only 52 melanoma examples. These are internal research measurements, not consumer-photo,
deployment, fairness, or medical-readiness evidence.

Cross-source experiments exposed a severe generalization ceiling:

- A frozen PAD-only ConvNeXt ensemble reached HIBA lesion macro F1 0.3388.
- PAD+HIBA full fine-tuning reached PAD macro F1 0.6270 and HIBA lesion macro F1 0.4590, failing all
  four preregistered cross-source gates.
- Validation-only partial freezing reached PAD macro F1 0.6032 and HIBA lesion macro F1 0.4487,
  again failing the four gate categories.
- ConvNeXt plus MedSigLIP distillation reached PAD macro F1 0.6345, MRA raw macro F1 0.4472, and
  paired macro F1 0.4801; it was rejected.

HIBA consists of clinician-taken smartphone imagery, and MRA-MIDAS uses standardized clinical
iPhone/iPad capture. Neither is a patient/consumer-taken holdout. DDI was rejected as a suitability
set because of clinical/procedural distribution differences and shortcut risks. No examined public
dataset establishes consumer smartphone generalization.

## Leakage and holdout caveats

- PAD-UFES appears in MedSigLIP's stated pretraining data, so PAD-UFES results involving MedSigLIP
  are not clean independent evidence.
- MedSigLIP access, size, terms, and deployment implications remain unresolved; it is not wired into
  the public demo.
- MILK10k remains sealed as a future holdout. It has not been inspected, tuned against, or used.

## Known risks

- Severe training-to-use-domain mismatch
- Sparse melanoma and weak SCC evidence in relevant phone-photo research
- Unknown behavior across skin tones, devices, lighting, framing, artifacts, and uncommon lesions
- Closed-set forcing: an out-of-vocabulary image still competes among known categories
- Uncalibrated softmax score and no established clinical threshold
- Potential harm from false reassurance, unnecessary alarm, or delayed professional care

## Prohibited interpretations

Do not claim or imply diagnosis, cancer detection, safety, clinical validity, fairness, medical
accuracy, consumer-phone readiness, deployment readiness, or superiority to professional review.
Passing software tests proves only software behavior, not medical validity.

## Promotion policy

No research candidate currently justifies replacing the demo model. A future candidate requires
lawfully usable patient/consumer smartphone data, reliable reference labels, patient/lesion grouping,
diverse devices/lighting/skin tones, sufficient melanoma and SCC support, a preregistered evaluation
protocol, and a genuinely untouched external holdout. Any production model change requires a
separate explicit approval.
