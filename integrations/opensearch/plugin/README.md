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

Команда запускает в Docker Checkstyle, clang-tidy, ShellCheck, rustfmt, Clippy и тест готового ZIP с `-Xcheck:jni`. Он открывает десять словарей в одном кеше, проверяет леммы и атрибуты Lucene для каждого языка, а также Unicode, рост буфера и закрытие словаря при одновременных запросах. Для проверки своего комплекта с распакованным OpenSearch на Linux выполните

```sh
./integrations/opensearch/plugin/test.sh /path/to/opensearch /path/to/dictionaries
```

## Анализ текста

В одном индексе настроены поля `text.rus`, `text.eng`, `text.deu`, `text.fra`, `text.spa`, `text.ita`, `text.por`, `text.ukr`, `text.jpn` и `text.yue`. Приложение знает язык документа и пишет текст в соответствующее поле. В документе с переводами можно заполнить несколько полей. У каждого поля свой анализатор и общий `bundle_path` со значением `udlex`. [Настройки одного индекса на десять языков](examples/index-settings.json) · [Как OpenSearch выбирает анализатор поля](https://docs.opensearch.org/latest/mappings/mapping-parameters/analyzer/)

Русская цепочка состоит из `standard → udlex(language=rus) → lowercase`. Фильтр сохраняет позиции и смещения, учитывает исходный регистр и пропускает токены с признаком `keyword`. Параметр `use_rules` управляет суффиксными правилами. [Настройки 224 анализаторов](../docker/analysis-224.json) · [Примеры запросов и подсветки](../docker/example.http)

Словарь остаётся открытым до остановки узла. Для смены словарей установите новый комплект в отдельный каталог под `config`, создайте индекс с новым `bundle_path`, переиндексируйте документы и переключите алиас. [Повторяемый бенчмарк](../../../benchmarks/opensearch/README.md)
