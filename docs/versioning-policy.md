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

Постоянный артефакт модели и вычислительный кеш имеют разные правила загрузки:

- Каталог модели (manifest, schema, model body, leaderboard), записанный
  `save_artifact` и читаемый `load_artifact`, имеет `ARTIFACT_VERSION=1`.
  Изменения формата требуют совместимости или миграции: модели предыдущих
  выпусков должны оставаться загружаемыми в пределах одного MAJOR.
  Неподдерживаемая версия артефакта вызывает явную ошибку.
- Candidate/stage cache имеет `CACHE_VERSION=2`; несовместимые записи
  считаются промахом и пересчитываются.
- HPO checkpoint имеет формат 2; версия, вычислительный контекст и история
  trials проверяются перед возобновлением. Несовместимая история пересчитывается.

Версии этих форматов независимы от версии пакета. Пересчёт кеша не отменяет
обязательство совместимости сохранённых моделей. Кеш загружают только из
доверенного каталога.
