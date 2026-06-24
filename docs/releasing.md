# Releasing

Pushing a `v*` tag triggers the release pipeline (`.github/workflows/release.yml`).
The distribution name is `honestml` (`[project]` in `pyproject.toml`);
`[project.urls]` points at `github.com/sukhov-is/HonestML`.

## Pipeline

- **check** — triple version gate `tag == pyproject == honestml.__version__`
  (`scripts/check_tag_version.py`), then verifies the tagged SHA is on `main`
  with a green CI run.
- **build** — builds the sdist and wheel.
- **audit** — installs the wheel with the `boosting` extra into a clean venv
  and runs pip-audit over it; a second pip-audit invocation in the same job
  emits a CycloneDX SBOM. The escape
  valve for a CVE without a released fix is `audits/pip-audit-ignore.txt`:
  every entry lands via PR with a justification, a review-by date, and a
  CHANGELOG line.
- **publish** — uploads to PyPI via trusted publishing (OIDC) from the `pypi`
  environment; attestations are generated automatically.
- **github-release** — creates the GitHub Release with auto-generated notes
  and attaches the distribution files and the SBOM.

## One-time setup (prerequisites)

1. **Trusted Publisher** on PyPI: register the repository with workflow file
   `release.yml` **and environment `pypi`** — the environment name is part of
   the trust anchor.
2. **Environment protection rules** for `pypi` in GitHub settings (required
   reviewers / tag-only deployment policy) — without them the environment is
   decorative.
3. **GitHub Pages** for the docs site: repository Settings → Pages → Source =
   "GitHub Actions". `docs-deploy.yml` then publishes the site (plus
   `llms.txt`/`llms-full.txt`) to `https://sukhov-is.github.io/HonestML/` on
   every push to `main` — the `Documentation` URL in `pyproject.toml`.

## Per-release checklist

1. Full suite green on `main` (the check job enforces this mechanically).
2. Green `workflow_dispatch` run of `benchmark.yml` **on the commit being
   tagged** — the gate is no regress vs `benchmarks/baseline.json`.
3. Bump the version in BOTH places: `pyproject.toml` and
   `honestml.__version__` (plus the pin in `tests/unit/test_public_api.py`).
4. Cut the release section out of `[Unreleased]` in `CHANGELOG.md`.
5. Tag `vX.Y.Z` and push; the pipeline does the rest.
6. Paste the benchmark run URL into the auto-created GitHub Release notes
   (the SBOM is attached by CI).

### First release

`benchmarks/baseline.json` does not exist until it is bootstrapped: dispatch
`benchmark.yml` with `update_baseline: true`, download the `benchmark-results`
artifact, and commit `baseline.json` together with a CHANGELOG line.
Subsequent releases gate against the committed baseline.
