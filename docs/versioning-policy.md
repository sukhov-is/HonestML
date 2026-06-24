# Versioning & deprecation policy

The library follows [Semantic Versioning 2.0.0](https://semver.org/): `MAJOR.MINOR.PATCH`.

## Public surface

The public API is the set of names re-exported from `honestml` and documented in the
[API reference](api.md), plus the domain ports in `honestml.core` and
`honestml.composition.registry.ComponentDescriptor` (together they are the contract
for third-party plugins — see the [plugin contract](plugin-contract.md); plugin
descriptors carry their own `api_version` integer, versioned separately from the
package). A test pins the top-level surface so additions and removals are deliberate.

## What each version part means

- **MAJOR** — an incompatible change to anything in the public surface (e.g. the `AutoML`
  facade and `FittedModel`, the config classes, the domain ports/Protocols, the exception
  hierarchy) or to a persisted format (below).
- **MINOR** — backward-compatible additions (new adapters, new metrics/splitters,
  new optional role-interfaces).
- **PATCH** — backward-compatible bug fixes.

## Deprecation

A deprecated public name or port method emits a `DeprecationWarning` for **at least
one minor release** before removal, with the replacement named in the message.
New capabilities are added through **new role-interfaces** (e.g. `SupportsShap`),
not by widening an existing base Protocol — so existing plugins keep working.

## Persisted formats

Two on-disk formats are covered by this policy:

- the model artifact directory (manifest, schema, model body, leaderboard) written by
  `save_artifact` and loaded as a `FittedModel` via `load_artifact`;
- the on-disk candidate cache (`cache_dir/<fingerprint>/<candidate_id>/`).

Changing a persisted format must be backward-compatible or come with a migration:
artifacts saved by an earlier release stay loadable within the same MAJOR series.
Both formats carry an integer version that gates loading — an artifact with an unsupported version is refused with a clear
error, and an incompatible cache entry is treated as a miss and recomputed.
