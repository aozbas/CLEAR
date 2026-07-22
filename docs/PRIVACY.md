# CLEAR Public Demo Privacy Boundary

## Scope

This document describes the tracked mobile and backend application code. It does not promise that an
unknown hosting, proxy, mobile-distribution, or network provider has privacy-safe defaults. Those
runtime facts must be verified before a public deployment.

## Data lifecycle

1. The user chooses or captures a photo. Expo ImagePicker may create an app-owned temporary cache
   file. CLEAR never deletes an original that already exists in the user's photo library.
2. The mobile app holds the selected URI and preview only for the request. Expo FileSystem uploads
   that file as binary content; the app does not create an account or application record.
3. The app sends the raw bytes over the configured API connection as `image/jpeg` or `image/png`.
   Production must use HTTPS. No filename, email, token, or account identifier is sent.
4. The backend rejects unsupported, empty, malformed, mismatched, oversized, or over-pixel-limit
   content while streaming and validating it in memory. The endpoint does not use multipart upload
   spooling or write an application temporary file.
5. The backend passes the validated bytes to the configured inference adapter, then returns one
   experimental result. It has no database, object-storage, analytics, or prediction-history client.
6. After the request, the app releases the preview and idempotently deletes the picker result only
   when its URI is inside CLEAR's app-cache directory. The JSON result remains in component memory
   until the user clears or leaves the screen. A process crash can leave an OS-managed cache file
   until the operating system purges it; deployment/device testing must verify this lifecycle.

The predictor is synchronous. If its request deadline expires or the client disconnects, the worker
may continue holding the image bytes in process memory until that call returns. CLEAR keeps the
bounded worker slot reserved during that interval so another request cannot multiply the exposure.
The code does not write those bytes to durable storage, but Python does not guarantee immediate
memory zeroization.

All API responses, including errors, receive `Cache-Control: no-store`, `Pragma: no-cache`,
`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
`Referrer-Policy: no-referrer`, and a restrictive camera/microphone/geolocation permissions policy.

## Threat and retention summary

| Area | Application guarantee | Deployment obligation |
| --- | --- | --- |
| Durable storage | No photo, result, user, scan, or history persistence code | Do not add request-body capture, object storage, database writes, or persistent volumes |
| Logs | Container default disables access logs; application errors use fixed messages and never log image bytes | Verify proxy/platform logs do not record bodies, authorization headers, or long-lived identifiers; minimize IP log retention |
| Transport | Client sends one raw image body to the configured backend | Require HTTPS with valid certificates; never deploy the HTTP development URL publicly |
| Third parties | No Hosted database, analytics, crash-reporting, advertising, or account SDK is used in the public flow | Audit hosting, DNS, TLS, app-store, and observability providers and their retention terms |
| Errors | Client receives fixed status-based copy without response bodies or trace details | Keep debug mode off and prevent infrastructure error pages from exposing internals |
| Identifiers | No account, email, filename, device ID, or app-generated request ID is sent | Network providers may still observe IP address and timing; document and minimize that processing |

## Operational acceptance checks

Before a public deployment, verify all of the following against the actual runtime:

- HTTPS redirects and HSTS are active; plaintext requests never carry images.
- Proxy and platform body limits are at most the backend limit.
- Access, WAF, APM, exception, and packet-capture settings do not retain request bodies.
- Log samples contain no image bytes, multipart data, filenames, tokens, or full headers.
- The container filesystem is read-only except for a bounded ephemeral `/tmp`.
- The service has no database/storage credentials and no outbound analytics integration.
- CORS and allowed-host lists contain only the intended deployed domains.
- Timeout, concurrency, and overload paths return the documented safe errors.
- Data-processing terms and retention for every infrastructure provider are documented.

If any runtime fact is unknown, describe the demo as stateless in application code—not as a blanket
guarantee that no network or infrastructure metadata exists.
