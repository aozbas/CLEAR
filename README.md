# CLEAR

CLEAR is an educational and technical skin-lesion image-classification project. Its public product
surface is a privacy-first, stateless demo: one image request produces one experimental outcome,
which may be a classification or an explicit abstention, with no account, image library, saved
result, or prediction history.

It is **not a medical device**. Its output is not a diagnosis, should not reassure or alarm anyone,
and must not be used for treatment or other medical decisions.

## Public demo boundary

The Expo mobile app sends a raw JPEG or PNG body to `POST /predictions/demo`. The FastAPI backend
validates the upload in memory, rejects conservatively defined unusable images before inference,
invokes the configured classifier for remaining inputs, and returns one JSON result. A cloud-verified
supported-input gate suppresses the category and score for rejected inputs. Application code does
not persist the submitted bytes or result. The app deletes its own temporary picker-cache result
after the request and never deletes an original from the user's photo library.

The public app and API intentionally have no authentication, user profiles, database client,
storage bucket, scan insert, or result-history route. See [the privacy boundary](docs/PRIVACY.md) for
the complete lifecycle and deployment caveats.

## Current model boundary

Static configuration points the demo at an owner-selected, six-class ConvNeXt-Tiny checkpoint fit
with source-balanced PAD-UFES and HIBA development data. The originating experiment failed all four
preregistered cross-source gate categories. The final fit creates a runnable artifact; it does not
create new independent performance evidence or establish reliable behavior on patient- or
consumer-taken photos. Details and prohibited interpretations are in the
[model evidence card](docs/MODEL_CARD.md).

Model weights, raw datasets, generated splits and reports, caches, credentials, and private workflow
records are deliberately not tracked.

## Repository layout

- `mobile/` — Expo + React Native stateless demo app
- `backend/` — FastAPI upload validation and experimental-classification API
- `ml/` — separate research, training, evaluation, and inference code
- `docker/` and `compose.yaml` — reproducible backend/runtime definitions
- `docs/PRIVACY.md` and `docs/MODEL_CARD.md` — public product evidence
- `docs/DATA_EVIDENCE.md` — consumer-photo dataset acceptance and current suitability results
- `docs/DEPLOYMENT.md` — gated HTTPS hosting and privacy-safe logging runbook

The mobile app talks only to the backend. The backend is the sole boundary allowed to call the
inference adapter.

## Contributing and governance

Bug reports, suggestions, documentation improvements, tests, and focused pull requests are welcome.
Start with [the contribution guide](CONTRIBUTING.md), which includes privacy boundaries, review
expectations, and the required [Developer Certificate of Origin](DCO) sign-off. The
[governance policy](GOVERNANCE.md) explains how decisions are made for the official project.

## License and ownership

Unless a file or directory states otherwise, CLEAR source code and original project documentation
are available under the [Mozilla Public License 2.0](LICENSE). Copyright (c) 2026 Alpaslan Ozbas and
CLEAR contributors. Contributors retain copyright in their contributions while licensing accepted
work under the applicable project license.

The code license does not grant rights to the CLEAR name, logo, or other project branding; see the
[trademark policy](TRADEMARKS.md). It also does not relicense datasets, dataset content, model
weights, checkpoints, generated artifacts, third-party software, or third-party assets. Those
materials remain subject to their own terms and must be identified separately before use or
distribution.

## Non-ML verification

These checks are designed not to import, construct, or invoke an ML model:

```powershell
python -m pip install -r backend/requirements-api.txt -r requirements-dev.txt
python -m unittest discover -s backend/tests
python -m ruff check backend
python -m ruff format --check backend

Set-Location mobile
npm ci
npm run typecheck
npm run format:check
npm run test:privacy
npm run audit:high
```

ML execution is intentionally excluded from local and GitHub-hosted verification. Project policy
requires a fresh authorized UCSD cloud pod for model construction, inference, training, evaluation,
or ML tests.

## Configuration and deployment

Copy the relevant `.env.example` file and set only non-secret public mobile values there. Backend
deployment must explicitly configure:

- `MODEL_PATH` and `MODEL_VERSION`
- `CORS_ORIGINS` as exact comma-separated origins (never `*` for a public deployment)
- `ALLOWED_HOSTS` with the deployed API hostname
- upload, pixel, concurrency, queue, and prediction time limits

The repository supplies a non-root backend container with no access log or server-identification
header. Local Compose makes its filesystem read-only with bounded ephemeral `/tmp`. A real deployment
must also terminate TLS, bound request bodies at the reverse proxy, avoid body/header capture in
observability products, set short log retention, and verify that the configured model checkpoint is
available through `/ready`. The [deployment runbook](docs/DEPLOYMENT.md) records the proposed managed
HTTPS design and the owner/provider approvals still required before it can go live.
