# udlex для OpenSearch 3.10.0 в Docker

## Зачем собирать образ

OpenSearch загружает плагины при старте узла. Ручная установка в файловый слой контейнера действует до его пересоздания. Чтобы udlex был в каждом новом контейнере, Docker берёт готовый образ OpenSearch 3.10.0 и добавляет два слоя. OpenSearch, плагин и словари уже собраны.

1. **Плагин и все 224 словаря.** Docker проверяет ZIP и устанавливает плагин для Linux AMD64 или ARM64 командой `opensearch-plugin install`. В `plugins/analysis-udlex/` появляются Java-плагин и нативная библиотека udlex. Установщик проверяет SHA-256 каждого языкового архива, распаковывает его в `config/udlex-224-20261120/` и сверяет версию словаря.
2. **Параметр Java.** В `config/jvm.options.d/90-udlex.options` добавляется строка `--enable-native-access=ALL-UNNAMED`, которая разрешает Java-плагину вызывать нативную библиотеку.

Пути указаны относительно `/usr/share/opensearch`. Образ собирают один раз для каждой архитектуры и повторно при обновлении базы, плагина или словарей. Готовый образ можно раздать другим узлам через свой реестр, указав в Compose только `image`. Язык поля задаётся в индексе. [Dockerfile комплекта](Dockerfile) · [Установка плагинов в Docker по документации OpenSearch](https://docs.opensearch.org/latest/install-and-configure/install-opensearch/docker/#working-with-plugins)

## Файлы для сборки

Комплект `udlex-compose-3.10.0-224.tar` содержит каталог `udlex/` со всеми входами Docker. При сборке из Git распакуйте комплект отдельно и скопируйте из `udlex/` файл `DICTIONARIES.json`, каталог `dictionaries/` и оба `plugins/*.zip` в соответствующие места рядом с этим Dockerfile. В Compose укажите этот каталог как `build.context`.

ZIP также можно [собрать из исходников](../plugin/README.md#сборка-из-исходников) под архитектуру целевых узлов. Для своей сборки пересчитайте `plugins/SHA256SUMS` по имеющимся ZIP командой `(cd plugins && sha256sum analysis-udlex-3.10.0-*.zip > SHA256SUMS)`. Установщик ожидает каталог словарей от 20 ноября 2026 года и проверяет его SHA-256.

## Установка в существующий Compose

В Dev Tools запросом `GET /` проверьте версию узла — нужна **3.10.0**. Рядом с действующим `compose.yaml` распакуйте архив и проверьте тома сервиса. Подставьте его имя из `docker compose config --services`.

```sh
tar -xf udlex-compose-3.10.0-224.tar
svc=opensearch
node=$(docker compose ps -q "$svc" | head -n 1)
docker inspect "$node" --format '{{.Config.Image}}{{range .Mounts}}{{println}}{{.Source}} -> {{.Destination}}{{end}}'
```

Индексы должны лежать на постоянном томе или в каталоге хоста, подключённом к `/usr/share/opensearch/data` (либо к вашему `path.data`). Если данные внутри контейнера, перенесите их на постоянное хранилище до пересоздания.

В существующем сервисе задайте `image` и `build`. По умолчанию используется официальный базовый образ. Если у вас свой образ с дополнительными плагинами, укажите в `OPENSEARCH_BASE` его прежний тег или digest.

```yaml
services:
  opensearch:
    image: local/opensearch-udlex:3.10.0-224
    build:
      context: ./udlex
      # Для собственного базового образа 3.10.0:
      # args:
      #   OPENSEARCH_BASE: registry.example.com/search/opensearch:3.10.0
```

**Если целиком подключён каталог `config/`**, он перекрывает конфигурацию образа. Добавьте словари и параметр Java в подключённый каталог. Пример для `./config` на хосте

```sh
mkdir -p ./config/jvm.options.d
python3 ./udlex/install-dictionaries.py ./config/udlex-224-20261120
cp ./udlex/90-udlex.options ./config/jvm.options.d/
```

Для именованного тома или своего `OPENSEARCH_PATH_CONF` используйте активный каталог конфигурации. Если отдельно подключены каталоги `plugins/` или `jvm.options.d/`, добавьте в них плагин или файл параметров соответственно. При подключении отдельных файлов, например `opensearch.yml` и сертификатов, достаточно файлов из образа.

```sh
docker compose build "$svc"
docker compose up -d --no-deps --pull never "$svc"
```

Первая команда готовит образ, пока узел обслуживает запросы. Вторая пересоздаёт контейнер из локального образа и подключает прежний том индексов. Единственный узел будет недоступен во время перезапуска. В кластере обновляйте узлы по одному, каждый раз дожидаясь возвращения узла и восстановления реплик. Для реплик одного Compose-сервиса используйте вашу процедуру поузлового обновления. [Docker Compose build](https://docs.docker.com/reference/cli/docker/compose/build/) · [Docker Compose up](https://docs.docker.com/reference/cli/docker/compose/up/)

При запуске через `docker run` соберите образ командой `docker build -t local/opensearch-udlex:3.10.0-224 ./udlex` и подставьте этот тег в вашу команду создания контейнера с прежними томами и настройками.

## Как включить лемматизацию

В Dev Tools выполните `GET /_cat/plugins?v` и найдите `analysis-udlex` на каждом узле. Проверочные запросы превращают «котами» в `кот`, а «cats» в `cat`. [Готовые запросы](example.http)

В новый индекс или его шаблон добавьте `settings.analysis` из файла ниже. В mapping задайте анализатор каждому языковому полю, например `"analyzer": "udlex_rus"`. Для существующих документов создайте индекс с новым mapping, выполните `_reindex`, учтите изменения и удаления за время копирования и после проверки переключите алиас. [Конфигурация 224 анализаторов](analysis-224.json) · [Пример полей](example.http) · [Переиндексация](https://docs.opensearch.org/latest/api-reference/document-apis/reindex/)

Сборка словарей от 20 ноября 2026 года включает 39 пакетов с некоммерческими условиями, отмеченных `-nc`. [Все языки и пакеты](LANGUAGES.txt)
