# Плагин udlex для OpenSearch 3.10.0

`analysis-udlex` добавляет в Lucene фильтр `udlex`. Он вызывает публичное Rust-ядро внутри узла OpenSearch. Например, при индексации слова «котами» и при поиске «котом» фильтр выдаёт один терм `кот`. Один ZIP работает в обоих официальных Docker-образах OpenSearch 3.10.0 для Linux AMD64 и ARM64. [Установка в существующий Compose со всеми 224 словарями](../docker/README.md)

## Сборка из исходников

Из корня репозитория выполните

```sh
./integrations/opensearch/plugin/build.sh
```

Та же сборка напрямую через Docker, в том числе из PowerShell в режиме Linux-контейнеров

```sh
docker build -f integrations/opensearch/plugin/Dockerfile --output type=local,dest=integrations/opensearch/plugin/dist .
```

На хосте нужен Docker с BuildKit. При первой сборке он скачивает инструменты и зависимости. Результат — `integrations/opensearch/plugin/dist/analysis-udlex-3.10.0.zip`. Внутри один Java-плагин и две нативные библиотеки. При запуске Java выбирает библиотеку своего процессора. В комплекте для Docker этот ZIP лежит рядом со всеми словарными архивами. [Java-фильтр](src/main/java/org/udlex/opensearch/) · [JNI](src/main/cpp/udlex_jni.cpp) · [Rust API](../../../rust/capi/src/lib.rs)

## Локальная проверка

```sh
./integrations/opensearch/plugin/lint.sh
```

Команда запускает в Docker Checkstyle, clang-tidy, ShellCheck, rustfmt, Clippy и тест готового ZIP с `-Xcheck:jni`. Тест проверяет Unicode, рост буфера, атрибуты Lucene и одновременные запросы при закрытии словаря. Для проверки своего комплекта с распакованным OpenSearch на Linux выполните

```sh
./integrations/opensearch/plugin/test.sh /path/to/opensearch /path/to/dictionaries rus котами кот
```

## Анализ текста

Для каждого языкового поля задайте анализатор, например `udlex_rus` или `udlex_eng`. Русская цепочка состоит из `standard → udlex(language=rus) → lowercase`. Фильтр сохраняет позиции и смещения, учитывает исходный регистр и пропускает токены с признаком `keyword`. Параметр `use_rules` управляет суффиксными правилами. [Настройки 224 анализаторов](../docker/analysis-224.json) · [Примеры запросов и подсветки](../docker/example.http)

Словарь остаётся открытым до остановки узла. Для смены словарей установите новый комплект в отдельный каталог под `config`, создайте индекс с новым `bundle_path`, переиндексируйте документы и переключите алиас. [Повторяемый бенчмарк](../../../benchmarks/opensearch/README.md)
