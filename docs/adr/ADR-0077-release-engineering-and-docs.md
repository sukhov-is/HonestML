# ADR-0077 — Релизная инженерия и документация

- **Статус:** Accepted
- **Драйверы:** DM-D5, DM-D6 (02-drivers.md); NFR-9/12/17/19 (G-L1/G-R1), R5

## Контекст

versioning-policy.md есть, процедуры релиза нет; LICENSE-файл отсутствует при
`license={text="MIT"}` (блокер публикации, G-L1); workflows — только ci.yml; README
устарел (описывает неактуальный API); mkdocs-скелет «filled in during M9»; plugin-contract.md
написан, но не в nav. Стандарт публикации 2026 — PyPI trusted publishing (OIDC),
attestations автоматически в `gh-action-pypi-publish` v1.11+.

## Решение

### §1. Лицензия и метаданные (G-L1)

`LICENSE` (MIT, «Copyright (c) 2026 AutoML contributors» — правка holder'а дёшева) в
корне; метаданные мигрируют на PEP 639: `license = "MIT"` (SPDX-строка) +
`license-files = ["LICENSE"]` (текущая table-форма `{text="MIT"}` deprecated и с
`license-files` не сочетается). URL-заглушки `github.com/honestml/honestml` и финальное
имя пакета на PyPI — **решение мейнтейнера перед первым релизом** (R-DLV-4);
фиксируются чек-листом релиза в `docs/releasing.md`, не выдумываются.

### §2. `release.yml` (минимальный, изолированный — security-модель PyPI)

Trigger `push: tags: v*`. Jobs:
- **check** — `scripts/check_tag_version.py` (чистая функция, вызывается и из
  workflow, и из юнит-теста): **тройная** сверка `tag == pyproject.version ==
  honestml.__version__` (в коде ДВА источника версии — pyproject.toml и хардкод в
  `honestml/__init__.py`, запинованный test_public_api; рассинхрон ловится здесь) +
  проверка «SHA тега достижим из main и его ci.yml-прогон successful» (gh api) —
  «релиз режется от зелёного main» становится механикой, не допущением.
- **build** — `python -m build` → артефакт dist/.
- **audit** — **объект: установленный wheel в чистом venv (+extras), не
  dev-окружение**; инструмент один: `pip-audit` (гейт) и он же
  `--format cyclonedx-json` (SBOM-артефакт) — ноль новых зависимостей. **Вентиль:**
  ignore-файл (vuln-ID + обоснование + срок пересмотра), общий для гейта и
  weekly-джобы; правки — через PR + строка в CHANGELOG (безфиксная транзитивная CVE
  не блокирует релизы намертво и не приводит к выпиливанию гейта).
- **publish** — `needs: [check, build, audit]`, `environment: pypi`,
  `permissions: id-token: write` ТОЛЬКО здесь,
  `pypa/gh-action-pypi-publish@release/v1` (attestations из коробки); затем GitHub
  Release с dist+SBOM.

Версия остаётся статической в pyproject (hatch-vcs отклонён). Prerequisites
(ручные, в releasing.md): регистрация Trusted Publisher на PyPI с указанием имени
workflow И environment; protection rules на environment `pypi` (иначе он
декоративен). Weekly-`schedule` pip-audit — **отдельный `audit.yml`** (не в ci.yml —
push/PR-триггеры не его); при красном прогоне джоба создаёт/обновляет GitHub issue
(сигнал не тонет).

### §3. Бенчмарк в релизном цикле

Релизный чек-лист включает зелёный `workflow_dispatch`-прогон `benchmark.yml`
(ADR-0076 §4) **на тегаемом коммите**, URL прогона фиксируется в GitHub Release
notes — пропуск становится видимым и аудируемым. Автоматики «тег ждёт бенчмарк» не
строим (минимализм; человек в цикле первого релиза). Туда же — пункт «срезать
секцию релиза из [Unreleased] в CHANGELOG до тега».

### §4. Документация (NFR-12, анти-дрейф DM-D6)

- **README** — переписывается под актуальный фасад: quickstart (`AutoML(...).fit`),
  таблица возможностей M0–M8 (honest selection/band, CV-схемы, budget/resume, FE/FS,
  HPO/ensemble, artifact+integrity, standalone inference, ONNX, tracking), extras-
  матрица, ссылка на docs. Неактуальный API не упоминается; **grep-тест** пинует отсутствие
  устаревших маркеров — именованная константа теста с комментарием-источником, скоуп
  строго README.md, word-boundary-матчи по списку маркеров в константе теста.
- **mkdocs nav**: + Quickstart, + API reference (**mkdocstrings[python]** в dev —
  генерация из докстрингов, не рукописная проза), + Plugin contract (существующий
  файл), + Correctness guide, + Releasing. `--strict` уже гейт. PEP 562-барели:
  API-ref опирается на `TYPE_CHECKING`-зеркала барелей (griffe собирает их статически
  как алиасы; инвариант синхронности трёх списков уже запинован
  `test_lazy_name_lists_in_sync`); fallback при нерезолве алиасов —
  `force_inspection: true` в mkdocstrings (лёгкий `import honestml` в docs-джобе
  допустим по дизайну).
- **Quickstart исполняем в CI** (NFR-DLV-6), авторский контракт: ВСЕ ```python-блоки
  quickstart самодостаточны в сумме (sklearn-данные, быстрые настройки) и
  исполняются последовательно в одном namespace; иллюстративное/неисполняемое —
  ```text-fence. Тест помечен `slow` — живёт в одной джобе (slow), НЕ в unit-матрице
  3 OS × 3 Python.
- **Correctness guide** — собирается ИЗ принятых ADR (honest selection/equivalence
  band, OOF-инвариант FE/FS, purge/embargo, outer holdout/finalize, artifact
  integrity + joblib-trust-model, serving-ограничения) и **known limitations** из
  re-triage хвостов: TEXT-роль не автоназначается; бустинги получают ordinal-коды
  (native categorical — future); early stopping не реализован; калибровка не входит
  в ONNX-граф; PII-замечание про artifact (категории/имена фич в манифесте — модель
  доверия документируется, G-P1-артефактная часть остаётся отдельным проходом).

### §5. Сопутствующая гигиена доков (re-triage хвостов)

`as-is-closure-status.md`: G-B1 → «FR-19, M9 (в работе/закрыт)»; G-L1 → закрыт;
G-R1 → закрыт release-джобой; G-D1/G-P1-артефакт — записи с явным статусом
known-limitation, владелец фиксации — этот срез (M9-3), исполнитель будущего фикса —
maintainer post-M9 re-triage. roadmap: битая ссылка «§0b» исправляется на §0a И
статус-таблица §0a обновляется (строка «M4–M9 ⛔ не начаты» давно лжёт — M0–M8
закрыты). `docs/implementation/changelog.md` не заводится — точка истины остаётся
корневой CHANGELOG.md (фиксируется здесь, чтобы вопрос не всплывал).

## Последствия

- (+) Публикуемость: лицензия, OIDC-публикация с attestations, SBOM/audit-гейт.
- (+) Docs перестают врать и защищены от дрейфа исполнимостью/grep/strict-гейтами.
- (−) mkdocstrings — новая dev-зависимость (только docs-job).
- (−) Имя PyPI/URLs — внешняя ручная точка (пользователь) до первого тега.
