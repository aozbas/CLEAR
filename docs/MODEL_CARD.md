# CLEAR Experimental Demo Model Evidence Card

## Intended scope

CLEAR is an educational and technical demonstration of image-classification plumbing. It returns a
single experimental category for a submitted skin-lesion image. It is not a medical device,
diagnostic system, triage tool, screening tool, clinical decision aid, or consumer reassurance tool.

## Model actually wired into the demo

Static backend and inference configuration currently selects:

- Architecture: fully fine-tuned ConvNeXt-Tiny
- Training domain: source-balanced PAD-UFES and HIBA development images
- Output vocabulary: actinic keratosis, basal cell carcinoma, melanoma, nevus, squamous cell
  carcinoma, and seborrheic keratosis
- Configured model-version string: `pad-hiba-convnext-tiny-source-balanced-final-2026-07-22`
- Display value: the top softmax output, labeled in the UI as an **uncalibrated model score**

The repository does not track the model weights. This wiring was determined by static code and
configuration inspection; this documentation does not load or evaluate the model.

## Training-data and checkpoint rights boundary

The [PAD-UFES-20 data article](https://doi.org/10.1016/j.dib.2020.106221) and associated data are
published under CC BY 4.0. The HIBA cohort comes from the
[Hospital Italiano de Buenos Aires collection](https://api.isic-archive.com/collections/251/); the
exact source metadata used by CLEAR was mechanically checked as CC-BY. ISIC requires collection and
image attribution to be retained.

Those terms are separate from CLEAR's source-code license, which does not relicense datasets or a
trained checkpoint. Before publicly hosting or distributing the configured checkpoint, the project
owner must record exact source provenance, verify that checkpoint distribution and the intended use
are permitted by both dataset sources, and retain all required notices and citations. The checkpoint
must remain untracked until that rights review is complete.

## Central limitation

PAD-UFES and HIBA are clinical/clinician-collected smartphone-image sources, not evidence of
patient- or consumer-taken-photo generalization. The source-balanced experiment failed its four
cross-source promotion-gate categories. Its score must not be interpreted as disease probability,
severity, urgency, or certainty.

## Development evidence, including failed gates

Research on PAD-UFES phone-photo data found ConvNeXt-Tiny to be the strongest internal candidate
(five-fold macro F1 0.6555; pooled macro F1 0.6609). SCC remained weak (F1 0.2444), and PAD-UFES had
only 52 melanoma examples. These are internal research measurements, not consumer-photo,
deployment, fairness, or medical-readiness evidence.

Cross-source experiments exposed a severe generalization ceiling:

- A frozen PAD-only ConvNeXt ensemble reached HIBA lesion macro F1 0.3388.
- PAD+HIBA full fine-tuning reached PAD macro F1 0.6270 and HIBA lesion macro F1 0.4590. Its source
  mean was 0.5430, worst source was 0.4590, and mean selected train-to-validation gap was 0.4273. It
  failed the gap, HIBA-overall, source-mean, and worst-source gate categories.
- Validation-only partial freezing reached PAD macro F1 0.6032 and HIBA lesion macro F1 0.4487,
  again failing the four gate categories.
- ConvNeXt plus MedSigLIP distillation reached PAD macro F1 0.6345, MRA raw macro F1 0.4472, and
  paired macro F1 0.4801; it was rejected.

HIBA consists of clinician-taken smartphone imagery, and MRA-MIDAS uses standardized clinical
iPhone/iPad capture. Neither is a patient/consumer-taken holdout. DDI was rejected as a suitability
set because of clinical/procedural distribution differences and shortcut risks. No examined public
dataset establishes consumer smartphone generalization.

The configured artifact is a fixed 11-epoch final fit on all 2,298 PAD-UFES images and 309 HIBA
images (308 lesions), using the original equal source/class weighting. Eleven is the median of the
five locked selected epochs (15, 8, 10, 11, 13). Because all approved development data participate
in this fit, it has no new independent validation metric. The owner selected it for the experimental
demo despite the failed gates; that product choice does not overturn the research result.

## Leakage and holdout caveats

- PAD-UFES appears in MedSigLIP's stated pretraining data, so PAD-UFES results involving MedSigLIP
  are not clean independent evidence.
- MedSigLIP access, size, terms, and deployment implications remain unresolved; it is not wired into
  the public demo.
- MILK10k remains sealed as a future holdout. It has not been inspected, tuned against, or used.

## Known risks

- Severe development-to-public-use-domain mismatch and failed cross-source gates
- Sparse melanoma and weak SCC evidence in relevant phone-photo research
- Unknown behavior across skin tones, devices, lighting, framing, artifacts, and uncommon lesions
- Closed-set forcing: an out-of-vocabulary image still competes among known categories
- Uncalibrated softmax score and no established clinical threshold
- Potential harm from false reassurance, unnecessary alarm, or delayed professional care

## Prohibited interpretations

Do not claim or imply diagnosis, cancer detection, safety, clinical validity, fairness, medical
accuracy, consumer-phone readiness, deployment readiness, or superiority to professional review.
Passing software tests proves only software behavior, not medical validity.

## Selection and future promotion policy

The configured model was selected by the project owner for the experimental demo despite not
meeting the research promotion gates. It must not be described as promoted, validated, or medically
better. A future evidence-based promotion requires lawfully usable patient/consumer smartphone
data, reliable reference labels, patient/lesion grouping, diverse devices/lighting/skin tones,
sufficient melanoma and SCC support, a preregistered evaluation protocol, and a genuinely untouched
external holdout. Any later production-model change requires separate explicit approval.
