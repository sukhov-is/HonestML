# ADR-0075 — Отчёт рана: markdown/HTML-рендер поверх run_report

- **Статус:** Accepted
- **Драйверы:** DM-D3 (02-drivers.md); FR-14/P13, F4.4/F4.5/F4.8

## Контекст

JSON-отчёт (ADR-0033) — единственный источник истины и уже запинован тестами как
аддитивно-эволюционирующий. Человеко-читаемой формы нет (известный пробел F4.4: «сводка
только при MLflow»). Extra `report=["matplotlib>=3.7"]` объявлена с M0 и не
используется. Сырых предсказаний в JSON нет — кривые калибровки/ROC невозможны без
расширения отчёта (Day-2).

## Варианты

1. **`render_report` в composition: markdown — stdlib-всегда, HTML — self-contained с
   matplotlib-графиками при наличии extra, деградация без него.** ✅
2. jinja2-шаблонизация (LightAutoML-стиль) — новая зависимость ради одного шаблона.
   Отклонено.
3. Рендер внутри `fit` (автозапись) — навязывает I/O и формат каждому рану; отчёт
   рана — opt-in (P13 «Reporting opt-in, ортогонален»). Отклонено.

## Решение

### §1. Поверхность

`honestml.render_report(report, path, *, fmt="md") -> Path` — новый публичный ленивый
символ (живёт в СУЩЕСТВУЮЩЕМ `composition/run_report.py` рядом с `save_run_report` —
один модуль про I/O-формы отчёта, без почти-одноимённого соседа; barrel +
`EXPECTED_PUBLIC`). `report` —
`Mapping` (сам `run_report_`) ИЛИ путь к `run_report.json` (round-trip с
`save_run_report`). `fmt`: `"md"` (дефолт — без зависимостей) | `"html"`. Возврат —
путь записанного файла (`run_report.md`/`run_report.html` при path-директории).

### §2. Контент (только из JSON) и два аддитивных ключа отчёта

Run-report-JSON сегодня НЕ несёт task/metric (они вне `RunConfig` — research §2):
рендерить их «из config» невозможно. Решение — **два аддитивных top-level ключа
`"task"` (kind) и `"metric"` (имя) в `build_run_report`** (как `holdout_score` в
M8c: пробел отчёта сам по себе, G-O1; `RUN_MANIFEST_VERSION` остаётся 1; фасад
передаёт значения kwargs'ами — направление зависимостей не меняется).

Шапка: task, metric, winner, **holdout_score**, honestml_version, run_fingerprint,
preset (ADR-0074 §3). Таблицы: leaderboard (model_id/score/rank, победитель
помечен), band (members/unstable/winner_by_tiebreak), budget
(mode/exhausted/skipped), significance, опциональные блоки feature_selection / hpo /
ensemble / serving / cache / native_routing (ADR-0095) / cv (period-CV диагностика, ADR-0096 §4) —
рендерятся только при наличии (None → «off»), timings,
свёрнутый резолвнутый config. Чтение — `.get` по известным ключам (отсутствие
task/metric в legacy-отчёте → «n/a»), незнакомые игнорируются (NFR-DLV-4).
**Markdown-экранирование** (R-DLV-5 в дефолтном формате): в md-ячейках `|` → `\|`,
угловые скобки нейтрализуются — имена моделей/фич приходят из пользовательских
данных и не должны ломать таблицы или проносить сырой HTML в GitHub-рендер.

### §3. HTML и графики

HTML = тот же контент + инлайн-CSS + графики matplotlib (**Agg**, импорт только в
html-ветке — ленивость extra как у mlflow/onnx) как base64-PNG: бар-чарт скоров
leaderboard, бар-чарт таймингов стадий. Без matplotlib — WARNING + HTML без графиков
(graceful, FR-DLV-3). **Все строковые значения экранируются `html.escape`, и
шаблон гарантирует, что пользовательские строки попадают ТОЛЬКО в текстовые узлы и
квотированные атрибуты** — никаких inline-JS/обработчиков/URL/CSS-контекстов для
данных (R-DLV-5; тест со «зловредным» именем фичи). Файл self-contained: без
внешних ссылок/CDN.

### §4. Чего НЕТ (Day-2 — operational.md)

Кривые калибровки/ROC/PR, feature-importance-графики — требуют сырых
предсказаний/важностей, которых в run_report нет; добавление = аддитивные ключи
отчёта отдельным проходом. Авто-вызов из `fit` — не планируется (opt-in).

## Последствия

- (+) F4.4 закрыта без новых обязательных зависимостей; extra `report` обретает смысл.
- (+) Рендерер — чистый потребитель: эволюция отчёта его не ломает (тестируется).
- (−) Поверхность: +`render_report` (запинована); два формата на поддержке.
