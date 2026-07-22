# Contributing To CLEAR

Thank you for helping improve CLEAR. Bug reports, implementation proposals, documentation fixes,
tests, and focused pull requests are welcome.

CLEAR is an educational and technical experimental-classification project. Contributions must not
describe it as a diagnostic, screening, triage, treatment, clinical, or consumer-reassurance tool.

## Before contributing

- Search existing issues and pull requests before opening a duplicate.
- Open an issue before a large feature, architecture change, dependency expansion, data-source
  addition, or change to model/inference behavior.
- Keep changes small enough to review and verify independently.
- Never submit real lesion photos, personal or health information, credentials, private datasets,
  model weights, checkpoints, generated reports, or service configuration containing secrets.
- Do not add persistence, accounts, authentication, scan history, image storage, analytics, or
  medical claims to the public demo without prior maintainer approval and a privacy review.

Security-sensitive reports should not include exploit-ready details, credentials, private user
information, or sensitive images in a public issue. Contact the project owner privately through the
address listed on their GitHub profile when responsible disclosure is necessary.

## Issues and suggestions

A useful issue includes:

- the problem or proposed improvement;
- why it fits the current educational, privacy-first demo scope;
- the relevant app, backend, ML, documentation, or infrastructure area;
- a minimal reproduction for bugs, using synthetic or non-sensitive inputs; and
- expected behavior and acceptance criteria.

Software-test results demonstrate software behavior only. They must not be presented as evidence of
medical validity, diagnosis accuracy, clinical readiness, or consumer-photo generalization.

## Pull requests

1. Fork the repository and create a short, descriptive branch.
2. Make a focused change that preserves the mobile/backend/ML boundaries.
3. Add or update tests when behavior changes. Do not include model execution in ordinary local or
   hosted checks unless the maintainers have explicitly approved the required isolated environment.
4. Run the relevant non-ML checks documented in the root README.
5. Explain what changed, why, user impact, privacy implications, and verification in the pull
   request description.
6. Sign off every commit under the Developer Certificate of Origin 1.1.

Sign off a commit with:

```text
git commit -s -m "Describe the change imperatively"
```

The sign-off records that you have the right to contribute the work under the license indicated in
the repository. Read the complete certificate in [DCO](DCO). A sign-off is not a copyright
assignment: contributors retain copyright in their contributions while licensing them under the
applicable project license.

## Licensing

Unless a file or directory states otherwise, contributions to repository source code and original
project documentation are accepted under the [Mozilla Public License 2.0](LICENSE). Do not submit
third-party material unless its license is compatible, its origin and license are documented, and
all required notices are preserved.

The source-code license does not license datasets, dataset content, model weights, checkpoints,
generated artifacts, third-party assets, or project trademarks. See the root README and
[trademark policy](TRADEMARKS.md) for those boundaries.
