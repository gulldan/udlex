# udlex для OpenSearch 3.10

`analysis-udlex` добавляет токен-фильтр `udlex`. Он заменяет словоформу её леммой внутри процесса OpenSearch. Например, по запросу «кот» можно найти «котами». На пути запроса работают Lucene, JNI и Rust. Словари читаются на CPU.

Сборка проверена на OpenSearch **3.10.0**, Lucene **10.6.0**, встроенном Temurin **25.0.5+8**, Linux **x86_64**. Дескриптор ZIP привязан к точной версии OpenSearch. Для иной версии или архитектуры соберите плагин с её дистрибутивом и проверьте в целевом окружении.

## Установка

Понадобятся ZIP плагина и по одному архиву словаря на каждый язык. Команды ниже рассчитаны на распакованный дистрибутив Linux. Укажите фактические пути к OpenSearch, его конфигурации и загруженным файлам. Используйте один ZIP и одинаковые версии словарей на узлах кластера.

```bash
os=/usr/share/opensearch
conf=${OPENSEARCH_PATH_CONF:-"$os/config"}
files=/absolute/path/to/downloads

"$os/bin/opensearch-plugin" install --batch \
  "file://$files/analysis-udlex-3.10.0-x86_64.zip"
mkdir -p "$conf/udlex-bundle"
tar -xzf "$files/dictionaries/udlex-dict-rus-1.0.0-bundle.tar.gz" \
  -C "$conf/udlex-bundle" --strip-components=1
```

В корне словарей должны появиться `.manifests/` и `rus/current`. Сохраните `LICENSES/` из языкового пакета. Дайте системному пользователю OpenSearch доступ на чтение файлов и проход по каталогам. Добавьте `--enable-native-access=ALL-UNNAMED` в параметры JVM, затем перезапустите узлы по очереди. При контейнерном развёртывании включите плагин в образ и предоставьте словари внутри каталога конфигурации. [Установка плагинов OpenSearch](https://docs.opensearch.org/latest/install-and-configure/plugins/)

## Индекс и проверка

В [index-settings.json](examples/index-settings.json) поле `text` использует `udlex_text` для индексации и поиска. Цепочка для русского языка — `standard → udlex(language=rus) → lowercase`. Для другого языка укажите его трёхбуквенный ISO-код. Для нескольких языков задайте отдельные анализаторы и поля. Язык для фильтра выбирается в конфигурации индекса.

Создайте тестовый индекс через `PUT /udlex-demo` с телом из этого файла. Для локального стенда с HTTP команда из корня репозитория выглядит так. Для рабочего кластера используйте его адрес, TLS и учётные данные.

```bash
curl --fail-with-body -X PUT http://127.0.0.1:9200/udlex-demo \
  -H 'Content-Type: application/json' \
  --data-binary @integrations/opensearch/plugin/examples/index-settings.json
```

В OpenSearch Dashboards Dev Tools выполните следующие запросы. Первый возвращает токен `кот`. Последний находит документ `1` и подсвечивает исходное слово `котами`.

```http
POST /udlex-demo/_analyze
{"field":"text","text":"котами"}

PUT /udlex-demo/_doc/1?refresh=true
{"text":"Играю с котами"}

GET /udlex-demo/_search
{"query":{"match":{"text":"кот"}},"highlight":{"fields":{"text":{}}}}
```

Фильтр сохраняет число токенов, их позиции и смещения, пропускает защищённые токены и применяет суффиксные правила по умолчанию. Параметр `use_rules: false` отключает догадки по суффиксам. Морфология получает исходный регистр, затем отдельный фильтр `lowercase` приводит полученные термы к нижнему регистру. Контекст предложения и пользовательские словари в этой цепочке отключены.

Словари остаются открытыми до остановки узла. При обновлении положите новый комплект в отдельный каталог под `config`, укажите его в `bundle_path` нового индекса и переиндексируйте существующие документы. После проверки можно переключить алиас. Аналогичная переиндексация нужна для старых полей при первом подключении фильтра.

## Сборка из исходников

Нужны полный дистрибутив OpenSearch с JDK, toolchain из `rust-toolchain.toml`, компилятор C++17 и `zip`. Из корня этой рабочей копии выполните

```bash
bash integrations/opensearch/plugin/build.sh "$os" 3.10.0
bash integrations/opensearch/plugin/test.sh "$os" "$conf/udlex-bundle" rus котами кот
```

Скрипт собирает публичное ядро `udlex-rs` и C ABI этой рабочей копии с зафиксированным Cargo.lock, компонует их с JNI и создаёт `integrations/opensearch/plugin/dist/analysis-udlex-3.10.0-x86_64.zip`. Нативный тест проверяет передачу UTF-16, расширение буфера, позиции токенов и одновременное использование и закрытие словаря.
