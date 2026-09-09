# billing-cli

CLI для выставления счетов: создание, расчёт налога, месячные отчёты,
экспорт в CSV, синхронизация с удалённым реестром.

## Установка

```
pip install -e .
```

## Использование

```
billing create --customer "ACME" --amount 100
billing report --month 2026-05
billing export --format csv --out invoices.csv
billing sync
```

## Валидация

Флаг `--strict-validation` включён по умолчанию: счета с пустым customer или
нулевой суммой отклоняются. Отключить: `--no-strict-validation`.

## Архитектура

Слои описаны в docs/adr/ADR-0001-layering.md: ядро (`billing/core`) не зависит
от хранилища и CLI.
