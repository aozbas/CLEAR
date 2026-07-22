# Consumer-Photo Evidence And Dataset Acceptance

## Current conclusion

CLEAR still has no public dataset that can establish six-class performance on patient- or
consumer-taken smartphone photos with reliable reference labels. SCIN is the first exact public
consumer-photo source accepted for metadata-level study, but its labels and rare-class support make
it unsuitable as a six-class performance holdout. No SCIN images have been downloaded or used for
training, tuning, or evaluation.

## SCIN metadata audit

The [Skin Condition Image Network](https://github.com/google-research-datasets/scin) contains
self-taken images voluntarily contributed by US adults through a mobile web flow with informed
consent. It is governed by the separate
[SCIN Data Use License](https://github.com/google-research-datasets/scin/blob/main/LICENSE), which
requires attribution and prohibits re-identification or re-linking.

On 2026-07-22, CLEAR acquired only the public version-1 metadata into an ignored local directory:

- 5,033 cases and 10,407 image references
- `scin_cases.csv` SHA-256:
  `7923ee9ba9775af413ca115c4587dc1bd32259d7936aa97fa80d88dd616ca3c3`
- `scin_labels.csv` SHA-256:
  `616ad03c24c304f5f7ef01b4116e4d3274f1c71c9b19a0e3d5620698b081a9fe`
- 3,061 cases with a parseable weighted dermatologist differential

The aggregate counts can be reproduced without opening an image or emitting a case identifier:

```powershell
python tools/audit_scin_metadata.py
```

The public labels are retrospective image-based differentials, not pathology- or
encounter-confirmed final references. The counts below are cases in which a CLEAR class appears
anywhere in the supplied differential, followed by cases in which it ties for the maximum supplied
weight:

| CLEAR class | Anywhere in differential | Tied for maximum weight |
| --- | ---: | ---: |
| Melanoma | 7 | 2 |
| Squamous cell carcinoma / in situ | 37 | 21 |
| Basal cell carcinoma | 21 | 16 |
| Actinic keratosis | 22 | 15 |
| Nevus / mole | 19 | 12 |
| Seborrheic or irritated seborrheic keratosis | 63 | 38 |

Most weighted labels concern allergic, inflammatory, or infectious conditions outside CLEAR's
closed vocabulary. SCIN can support a separately preregistered consumer-domain/input-compatibility
study, but not a claim that the six-class category is correct. Any image use would require a fresh
UCSD pod, case-grouped analysis, the license attribution above, and a protocol frozen before images
are accessed.

## Dataset acceptance checklist

A candidate performance dataset must pass every hard gate before download or compute:

1. **Capture provenance:** the photographer is the patient/consumer or an explicitly separable
   non-clinician cohort using ordinary smartphones without dermatoscope attachments.
2. **Reference standard:** a final pathology, encounter-confirmed specialist reference, or another
   prespecified defensible reference is available. Image-only differentials are insufficient for
   the six-class performance question.
3. **Grouping:** stable patient and lesion identifiers allow leakage-free grouped splits and prevent
   multiple views from crossing partitions.
4. **Class support:** all six target classes are mapped without speculative relabeling, with enough
   melanoma and SCC lesions to report uncertainty rather than a misleading point estimate.
5. **Capture diversity:** devices, lighting, distance, focus, anatomy, skin tones, and acquisition
   settings are documented and materially varied.
6. **Shortcut audit:** rulers, markers, procedure fields, ex-vivo tissue, dermoscopy, duplicate
   images, text overlays, and source-specific artifacts can be measured or excluded mechanically.
7. **Consent and privacy:** research consent covers the proposed use, faces/identifiers are handled,
   re-identification is prohibited, and raw content can remain on authorized private systems.
8. **Rights:** download, analysis, derivative artifacts, aggregate reporting, model training, hosted
   inference, and any checkpoint distribution are each permitted or separately gated.
9. **Holdout discipline:** a genuinely untouched patient/lesion-grouped external partition can be
   sealed before candidate selection. MILK10k remains sealed and is not part of intake auditing.

## Candidate disposition

| Source | Capture fit | Reference fit | Current use decision |
| --- | --- | --- | --- |
| SCIN | Exact self-contributed consumer domain | Dermatologist differentials; sparse target classes | Metadata accepted; images deferred; domain-only research candidate |
| PAD-UFES-20 | Smartphone clinical images | Useful source references and grouping | Training/development source, not consumer evidence |
| HIBA | Clinician-taken smartphone images | Useful lesion/patient metadata | Training/development source, not consumer evidence |
| MRA-MIDAS | Standardized clinical iPhone/iPad capture | Stronger clinical references | Mismatched capture; not a consumer holdout |
| DDI | Clinical/procedural mixture | Pathology-confirmed | Rejected for consumer suitability and shortcut risk |
| SLICE-3D | Crops from standardized 3D total-body photography | Patient grouping and lesion labels | Rejected as patient-capture evidence; DSLR/standardized source |

## Lawful acquisition path

The highest-value next step is not another scrape. It is a data-sharing or prospective collection
agreement with a teledermatology program that can supply patient-taken original smartphone images,
patient/lesion identifiers, final encounter or pathology references, consent for research use, and
enough melanoma and SCC cases. Before contact or collection, prepare a one-page cohort specification,
data dictionary, minimum per-class targets with confidence-interval rationale, privacy/security plan,
and proposed data-use terms. Outreach, accepting restricted terms, or beginning human-subjects
collection requires separate owner and institutional approval.
