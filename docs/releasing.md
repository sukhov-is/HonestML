# Выпуск и публикация

Сайт документации и пакет honestml публикуются независимыми workflows. Сайт размещён на [GitHub Pages](https://sukhov-is.github.io/HonestML/); исходники и workflows — в [репозитории HonestML](https://github.com/sukhov-is/HonestML).

## Сайт документации

Push в main запускает docs-deploy.yml. Workflow устанавливает зависимости документации, выполняет `mkdocs build --strict`, создаёт `llms.txt` и `llms-full.txt` скриптом `scripts/build_llms_txt.py`, затем публикует site через GitHub Pages. Доступен также ручной workflow_dispatch. Публикация сайта сама по себе не выпускает пакет PyPI и не подтверждает прохождение CI библиотеки.

В настройках Pages источником должны быть GitHub Actions. Публичные страницы перечислены в mkdocs.yml; внутренние рабочие журналы и пакеты проектирования исключены из сайта. Ссылки на результаты должны вести к доступным читателю сводкам, а не к локальным журналам.

## Пакет PyPI

Push тега vX.Y.Z запускает release.yml:

1. Check запускает scripts/check_tag_version.py для сверки тега, версии pyproject.toml и honestml.__version__, затем проверяет принадлежность коммита main и успешный CI на этом SHA.
2. Build создаёт sdist и wheel.
3. Audit устанавливает wheel с boosting extra в чистое окружение, выполняет pip-audit и создаёт CycloneDX SBOM. Исключения уязвимостей учитываются в audits/pip-audit-ignore.txt с обоснованием, сроком пересмотра и записью CHANGELOG.
4. Publish загружает пакет на PyPI через OIDC в environment pypi с attestations.
5. GitHub-release создаёт релиз и прикладывает дистрибутивы и SBOM.

Trusted Publisher на PyPI должен указывать этот репозиторий, release.yml и environment pypi. Правила защиты environment (protection rules) задают требуемых reviewers и ограничения публикации. Эти настройки проверяются отдельно от содержимого workflow.

## Последовательность выпуска

1. Согласовать версию по [политике версионирования](versioning-policy.md). Обновить pyproject.toml, honestml.__version__, проверку версии в tests/unit/test_public_api.py и согласованный lockfile. Оформить секцию релиза из Unreleased в CHANGELOG.
2. Проверить готовое дерево, закоммитить изменения и отправить коммит в main. Дождаться успешного CI на точном SHA выпуска.
3. Запустить benchmark.yml через workflow_dispatch на этом же коммите с update_baseline=false. Требуется успешная проверка против benchmarks/baseline.json; результаты другой ревизии не заменяют её.
4. Создать и отправить тег vX.Y.Z на проверенный коммит. Проверить завершение release.yml и доступность нужной версии на PyPI.
5. Добавить ссылку на успешный benchmark run в заметки созданного GitHub Release и проверить приложенные дистрибутивы и SBOM.

Изменение baseline — отдельная содержательная работа: update_baseline=true создаёт артефакт benchmark-results, который нужно проверить и закоммитить вместе с обоснованием в CHANGELOG. Перезапись эталона не заменяет успешную проверку регрессий.
