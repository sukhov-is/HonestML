# Диаграммы: C4, DFD, граф зависимостей (Mermaid)

Все диаграммы — в Mermaid внутри markdown, чтобы рендерились в репозитории.
Минимум визуального шума, максимум смысла. Подписи — на русском.

## C4 — уровни

C4 описывает систему сверху вниз. Рисуй ровно столько уровней, сколько нужно
для понимания; чаще всего достаточно Context + Container.

### Уровень 1 — System Context (кто и зачем пользуется системой)
```mermaid
flowchart TB
    user[Пользователь / data-scientist]
    sys[["HonestML<br/>(наша библиотека)"]]
    src[CSV / parquet / DataFrame]
    out[Честный лидерборд + ModelArtifact]
    user -->|данные + конфиг прогона| sys
    sys -->|читает через Reader| src
    sys -->|оценка и победитель| out
```

### Уровень 2 — Container (пакеты библиотеки и то, что она пишет на диск)
```mermaid
flowchart TB
    subgraph honestml
        comp[composition<br/>facade.fit + build]
        app[application<br/>run_slice, скоринг, отчёт]
        ad[adapters<br/>Reader, сплиттеры, модели, сериализаторы]
        core[core<br/>сущности + порты]
    end
    files[(Файлы: ModelArtifact,<br/>кэш кандидатов, манифест прогона)]

    comp --> app
    comp --> ad
    app --> core
    ad --> core
    ad -->|save / load| files
```

### Уровень 3 — Component (модули внутри пакета; по необходимости)
```mermaid
flowchart LR
    build[build._resolve_splitter] -->|строит| splitter[TimeSeriesSplitter]
    slice[run_slice] --> port[[Port: CVSplitter]]
    splitter -.реализует.-> port
```

Уровень 4 (код) обычно не рисуют — его роль выполняет сам код и граф зависимостей.

## DFD — Data Flow Diagram (потоки данных)

Показывает, как данные текут между процессами и хранилищами. Полезно для
пайплайнов: чтение и трансформации признаков, нарезка фолдов, путь инференса.

```mermaid
flowchart LR
    src([CSV / parquet / DataFrame]):::ext
    p1[/Reader: валидация, роли колонок, категории/]:::proc
    ds[(PolarsDataset)]:::store
    p2[/Нарезка фолдов + OOF-обучение кандидатов/]:::proc
    p3[/Скоринг, значимость, выбор победителя/]:::proc
    out[(Лидерборд + ModelArtifact + манифест прогона)]:::store

    src --> p1 --> ds --> p2 --> p3 --> out
    classDef proc fill:#eef,stroke:#557;
    classDef store fill:#efe,stroke:#575;
    classDef ext fill:#fee,stroke:#755;
```

Обозначения: `([внешний источник])`, `[/процесс/]`, `[(хранилище)]`.

## Граф зависимостей + слои Clean Architecture

Ключевая проверка: **все стрелки направлены внутрь, к домену**. Если хоть одна
наружу — это нарушение правила зависимостей, отметь его явно.

```mermaid
flowchart TB
    subgraph FD["Frameworks & Drivers"]
        skl[sklearn]
        boost[CatBoost / LightGBM / XGBoost]
        pl[polars]
    end
    subgraph IA["Interface Adapters"]
        reader[Reader]
        splitter[KFoldSplitter]
        est[Обёртка бустинга]
    end
    subgraph UC["Use Cases"]
        slice[run_slice]
    end
    subgraph EN["Entities"]
        model[Dataset / Fold]
        port[[Port: CVSplitter]]
    end

    slice --> port
    slice --> model
    splitter -.реализует.-> port
    splitter --> pl
    reader --> pl
    est --> skl
    est --> boost
```

Читать так: `run_slice` (use-case) зависит от порта `CVSplitter`, а
`KFoldSplitter` (адаптер) порт **реализует** — стрелка зависимости развёрнута
внутрь через DIP. Порты живут во внутреннем слое (`honestml.core.ports`),
связывание — в `honestml.composition`.

Нарушение (стрелка наружу или цикл) помечай отдельным классом, чтобы оно бросалось
в глаза на ревью: `classDef violation stroke:#c00,stroke-width:2px;` и применяй его
к нарушающему узлу через `узел:::violation`.

## Когда какую диаграмму

- **C4 Context/Container** — почти всегда, даёт общую карту.
- **C4 Component** — если внутри пакета нетривиальная структура.
- **DFD** — для пайплайнов данных (чтение и трансформации признаков, нарезка
  фолдов, путь инференса).
- **Граф зависимостей** — всегда при проектировании на Clean Architecture:
  он визуализирует правило зависимостей; доказывает правило дельта контрактов
  import-linter, которой артефакт завершается (фаза 6 SKILL).
