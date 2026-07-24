# CLEAR Experimental Demo Model Evidence Card

## Intended scope

CLEAR is an educational and technical demonstration of image-classification plumbing. It returns a
single experimental outcome for a submitted image: either a six-class category or an explicit
abstention. It is not a medical device, diagnostic system, triage tool, screening tool, clinical
decision aid, or consumer reassurance tool.

## Model actually wired into the demo

Static backend and inference configuration currently selects:

- Architecture: fully fine-tuned ConvNeXt-Tiny
- Training domain: source-balanced PAD-UFES and HIBA development images
- Output vocabulary: actinic keratosis, basal cell carcinoma, melanoma, nevus, squamous cell
  carcinoma, and seborrheic keratosis
- Configured model-version string: `pad-hiba-convnext-tiny-source-balanced-final-2026-07-22`
- Display value: the top softmax output, labeled in the UI as an **uncalibrated model score**

The repository does not track the model weights. The configured file is 111,376,483 bytes with
SHA-256 `12c7261b06e3da9d1639e5e2c11220837de5a69f972acf25a55c4a0ae31d99b8`.
The final-fit manifest fingerprint is
`23d3f41f18fc6d1082434fc049b6a7b7af07785df70b668bfb5ec51115747c5d`.
These identities were checked without loading or evaluating the model.

## Training-data and checkpoint rights boundary

The [PAD-UFES-20 data article](https://doi.org/10.1016/j.dib.2020.106221) and associated data are
published under CC BY 4.0. The HIBA cohort comes from the
[Hospital Italiano de Buenos Aires collection](https://api.isic-archive.com/collections/251/); the
exact source metadata used by CLEAR was mechanically checked as CC-BY. ISIC requires collection and
image attribution to be retained.

Those terms allow sharing and adaptation when attribution, a license link, and change notices are
retained. They do not, by themselves, prohibit distributing a trained checkpoint. They are separate
from CLEAR's source-code license, which does not automatically relicense datasets or a trained
checkpoint.

The final fit was initialized from TorchVision ConvNeXt-Tiny `IMAGENET1K_V1`. TorchVision source code
is BSD-3-Clause, but [ImageNet's access terms](https://image-net.org/accessagreement) describe the
database as available only for non-commercial research and educational use, and
[ImageNet states](https://www.image-net.org/about.php) that it does not own the underlying image
copyrights. [TorchVision's model documentation](https://docs.pytorch.org/vision/stable/models.html)
also warns that pretrained models may carry licenses or terms derived from their training data and
places responsibility on the user to determine whether a use is permitted. The available terms do
not clearly resolve whether those restrictions carry into this fine-tuned checkpoint.

The resulting project decision is:

- **Public checkpoint download or redistribution: temporarily on hold.** The file remains ignored,
  untracked, and absent from public container images and release assets while its upstream
  pretrained-weight chain is reviewed.
- **Private server-side provisioning: conditionally allowed only for CLEAR's non-commercial
  educational experiment**, with the artifact in access-controlled storage and the PAD-UFES, HIBA,
  TorchVision, and ImageNet notices retained here.
- **Commercial use, relicensing, or claiming an unrestricted model license: blocked** until
  qualified legal review, written upstream permission, or replacement with a training chain whose
  data and pretrained-weight rights are unambiguous.

This is a conservative project distribution decision, not legal advice, a finding of illegality, or
a claim that model weights are legally an adaptation of any particular image. If distribution is
cleared later, the checkpoint needs its own artifact notice, dataset and upstream attributions,
scope/limitations, checksum, and applicable terms; CLEAR's MPL-2.0 code license alone is not enough.
At 111,376,483 bytes, the current file also exceeds
[GitHub's 100 MiB normal Git object limit](https://docs.github.com/repositories/working-with-files/managing-large-files/about-large-files-on-github),
so a cleared artifact would need an appropriate large-file or release channel rather than a normal
Git commit. The deployment gate is documented in [the backend deployment runbook](DEPLOYMENT.md).

### What a Hugging Face release would and would not solve

A Hugging Face model repository is technically well suited to this file size: Hub repositories are
versioned and use a large-file storage backend, and a client can download one file at an immutable
commit revision. A public repository can be downloaded without requiring every demo user to create
an account. The project could pin the exact revision and independently verify the artifact's
SHA-256 before the backend starts.

Hugging Face is a distribution channel, not a rights clearance service. Its
[terms for user content](https://huggingface.co/terms-of-service) require the uploader to have the
right to post the content and grant broad reuse rights when a repository is public. A
[gated model](https://huggingface.co/docs/hub/en/models-gated) still distributes files to approved,
authenticated users; gating collects user identity and does not create permission that the
uploader lacks. A private repository avoids public download but requires credentials and therefore
does not meet CLEAR's intended no-account contributor quickstart.

If a checkpoint with an unambiguous rights chain is approved later, the preferred bootstrap is:

1. publish the checkpoint and complete evidence card in a dedicated public model repository;
2. pin a full immutable Hub revision, filename, byte count, and SHA-256 in CLEAR;
3. let a one-shot Compose service download and verify the file into a named Docker volume;
4. start the backend only after that verifier exits successfully, mounting the file read-only; and
5. reuse the verified volume on later starts while continuing to fail closed on any identity drift.

This design keeps the source repository and backend image small, makes the first
`docker compose up --build` perform the only download, and avoids embedding tokens. It must not be
wired to the current checkpoint unless its upstream rights are resolved. The current repository
therefore continues to require the separately provisioned local file.

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
dataset establishes consumer smartphone generalization. A metadata-only audit of the exact
consumer-contributed SCIN source and the dataset acceptance gates are recorded in
[the consumer-photo evidence note](DATA_EVIDENCE.md).

The configured artifact is a fixed 11-epoch final fit on all 2,298 PAD-UFES images and 309 HIBA
images (308 lesions), using the original equal source/class weighting. Eleven is the median of the
five locked selected epochs (15, 8, 10, 11, 13). Because all approved development data participate
in this fit, it has no new independent validation metric. The owner selected it for the experimental
demo despite the failed gates; that product choice does not overturn the research result.

## Supported-input abstention evidence

The demo adds a post-hoc input-compatibility gate without changing the classifier weights. Three
scores from the fixed logits were preregistered: maximum softmax output, maximum logit, and
log-sum-exp. Calibration selected log-sum-exp with threshold `4.4970903396606445`. The disjoint
evaluation partition retained 95.80% of PAD-UFES and 98.60% of HIBA development images while
accepting 0 of 400 obvious non-skin Open Images examples across vehicles, animals, food, household,
and outdoor-object groups. Every frozen acceptance rule passed.

This result is deliberately narrow. The final classifier had already seen every PAD-UFES/HIBA
positive image during fitting, so positive retention is development behavior rather than independent
classifier evidence. The negative evaluation uses fixed, attributed Open Images categories and does
not establish detection of unsupported skin conditions, arbitrary/adversarial inputs, or reliable
consumer-photo behavior. Rejection means only that no classification is shown under this
experimental gate; it does not determine whether an image contains a lesion.

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
- Partial closed-set protection only: the abstention gate covers tested obvious non-skin categories,
  not every out-of-vocabulary condition or image
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
