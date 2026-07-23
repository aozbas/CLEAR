# CLEAR Experimental Backend Deployment Gate

## Current status

The contributor demo runs locally: the verified Windows path uses the pinned host-backend launcher
for the checkpoint, while Expo Go connects from a phone on the same trusted LAN. It does not require
a cloud service. Docker remains available for container verification and for hosts whose published
port is actually reachable over the LAN; on the tested Windows Docker Desktop configuration, port
8000 was reachable through loopback but not through the Wi-Fi interface. See the root README for the
preflight and launcher flow.

The repository is also prepared for a possible managed Cloud Run deployment, but no public service
has been created. The configured model has not been executed outside the authorized UCSD research
cluster. A future public launch remains gated on an owner-selected Google Cloud project and billing
account, an exact public origin, and explicit approval to run inference on Cloud Run rather than a
fresh UCSD pod.

This design does not add accounts, a database, analytics, image storage, or prediction history. The
Cloud Storage bucket described below contains only the private model artifact; user uploads never
enter it.

## Why this target

Cloud Run supplies a managed HTTPS endpoint and redirects plaintext HTTP before requests reach the
container. It can scale to zero and cap both concurrency and instance count. A private Cloud Storage
bucket can be mounted read-only, so the checkpoint stays outside the source repository and container
image. Cloud Run nevertheless creates platform request logs before Cloud Logging exclusions are
applied; the provider can observe network metadata even when CLEAR stores no image or result.

Authoritative operational references:

- [HTTPS behavior](https://cloud.google.com/run/docs/triggering/https-request)
- [Cloud Storage read-only mounts](https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts)
- [Cloud Run request logging](https://cloud.google.com/run/docs/logging)
- [Cloud Logging exclusions](https://cloud.google.com/logging/docs/routing/overview)
- [Public access configuration](https://cloud.google.com/run/docs/authenticating/public)

## Required launch decisions

Do not begin deployment until all of these are recorded:

1. The owner approves Cloud Run as an exception to the current UCSD-only model-execution boundary.
2. The owner selects the Google Cloud project, billing account, region, and exact web origin or
   confirms that the first release is native-mobile only.
3. The service remains a non-commercial educational experiment. Commercial use or public checkpoint
   download remains blocked by the upstream-rights decision in the model evidence card.
4. A budget alert is active, and the service is capped at one instance with one request at a time.
5. The checkpoint is uploaded only to a private bucket with public-access prevention and uniform
   bucket-level access.
6. The request-log exclusion and one-day default-log retention are verified before public access is
   enabled.

## Build without the checkpoint

Install and authenticate the Google Cloud CLI, select the intended project, and enable Cloud Run,
Cloud Build, Artifact Registry, Cloud Logging, and Cloud Storage. Create a private Docker repository,
then submit the build from the repository root:

```bash
gcloud builds submit . \
  --config deploy/cloud-run/cloudbuild.yaml \
  --substitutions _REGION=REGION,_REPOSITORY=REPOSITORY,_TAG=GIT_SHA
```

The explicit `.gcloudignore` and `.dockerignore` exclude private data, local environments, reports,
credentials, and every model artifact. Record the resulting immutable image digest; do not deploy a
mutable tag without resolving it to that digest.

## Provision the private model artifact

Verify the local file before any upload:

```powershell
(Get-FileHash `
  ml/models/pad_hiba_convnext_tiny_source_balanced_final_seed42.pt `
  -Algorithm SHA256).Hash.ToLowerInvariant()
```

The required value is
`12c7261b06e3da9d1639e5e2c11220837de5a69f972acf25a55c4a0ae31d99b8`.

Create a dedicated bucket with uniform access and public-access prevention. Disable object
versioning and use a dedicated service identity that has only `storage.objects.get` on this bucket.
Upload the exact checkpoint under its fixed filename, verify its remote hash, and never put user
uploads in this bucket.

## Deploy privately first

Copy `deploy/cloud-run/env.example.yaml` to an untracked operational file. Keep
`ALLOWED_HOSTS=localhost` for the first private revision, use the exact intended HTTPS origin for
`CORS_ORIGINS`, and deploy with these minimum controls:

- immutable container-image digest
- port 8000, 2 vCPU, 4 GiB memory
- request timeout 35 seconds
- concurrency 1, maximum instances 1, minimum instances 0
- dedicated service account with read-only checkpoint-bucket access
- read-only Cloud Storage mount at `/models`, with FUSE logging disabled
- unauthenticated access disabled
- TCP startup probe on port 8000

Retrieve the generated HTTPS service URL, replace `ALLOWED_HOSTS` with its exact hostname, and deploy
a second private revision. `/health` must return 200, and `/ready` must return 200 with
`model_checkpoint_present: true`. The readiness route checks only the file path and does not load the
model.

## Privacy-safe platform logging

Before public access, add an exclusion to the project `_Default` sink for this service's
`run.googleapis.com/requests` log and set `_Default` retention to one day. Keep required audit logs;
do not disable them. The exclusion must be scoped to `resource.type="cloud_run_revision"` and the
exact service name. Verify that it is enabled rather than merely present.

The container already disables Uvicorn access logs and its server-identification header.
Application warnings and errors use fixed text and do not include request bodies, filenames,
headers, labels, scores, or tracebacks. Retain only the minimum container/system logs needed to know
whether the service starts. Do not add APM, crash reporting, packet capture, request sampling, or a
WAF rule that stores bodies or headers.

Cloud Logging exclusions are applied after entries reach the Logging API. Therefore public privacy
copy must continue to say that CLEAR does not persist uploads or results in application code; it
must not promise that the hosting/network provider observes no IP address, timing, user agent, or
request metadata.

## Public cutover and verification

Public access is the final reversible switch. Enable it only after the private checks above and then:

1. Confirm HTTP redirects to HTTPS and every HTTPS response carries
   `Strict-Transport-Security: max-age=31536000` and `Cache-Control: no-store`.
2. Confirm the backend stops reading and returns 413 beyond 8 MiB. Cloud Run's outer request ceiling
   is larger and does not replace this application cutoff.
3. Send malformed, wrong-type, oversized, busy, and timeout requests and verify fixed safe errors.
4. With separate approval for non-UCSD inference, send one authorized synthetic or public test image
   and verify one experimental result or abstention, with no stored object or prediction record.
5. Verify request-log queries return no retained request entries and container logs contain no image
   bytes, filenames, full headers, identifiers, labels, or scores.
6. Set the production mobile `EXPO_PUBLIC_API_URL` to the HTTPS endpoint and confirm the HTTP default
   is absent from the release build.

If a check fails, make the service private immediately. Before deleting any cloud resource, list and
verify its exact project, region, service, bucket, and repository name. Cloud Run, Artifact Registry,
Cloud Storage, and any build staging bucket are separate billable resources and must be handled
explicitly.
