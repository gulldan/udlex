# udlex для существующего OpenSearch 3.10.0 в Docker

Плагин устанавливается **внутрь образа**. Docker берёт ваш готовый OpenSearch 3.10.0, вызывает в нём `bin/opensearch-plugin install`, добавляет все 224 словаря и параметр JVM для вызова Rust. При следующем старте OpenSearch загружает плагин. Один ZIP подходит для Intel/AMD и ARM. [Dockerfile](Dockerfile) · [Официальная инструкция OpenSearch](https://docs.opensearch.org/latest/install-and-configure/install-opensearch/docker/#working-with-plugins)

## Установка в Compose

1. Проверьте версию узла запросом `GET /`. Рядом с действующим `compose.yaml` распакуйте комплект. Подставьте имя своего сервиса из `docker compose config --services`.

   ```sh
   tar -xf udlex-compose-3.10.0-224.tar
   svc=opensearch
   node=$(docker compose ps -q "$svc" | head -n 1)
   docker inspect "$node" --format '{{.Config.Image}}{{range .Mounts}}{{println}}{{.Source}} -> {{.Destination}}{{end}}'
   ```

   Проверьте, что `/usr/share/opensearch/data` или ваш `path.data` находится на постоянном томе. Если индексы лежат в файловом слое контейнера, перенесите их на том до пересоздания.

2. В существующем сервисе задайте образ и каталог сборки. Сохраните настройки, тома и сертификаты сервиса. Если используете собственный базовый образ, укажите его прежний тег или digest в `build.args.OPENSEARCH_BASE`.

   ```yaml
   services:
     opensearch:
       image: local/opensearch-udlex:3.10.0-224
       build:
         context: ./udlex
   ```

   Если целиком монтируете `config/`, этот том перекрывает файлы образа. Добавьте в активный каталог конфигурации словари и параметр JVM. Пример для `./config` на хосте

   ```sh
   mkdir -p ./config/jvm.options.d
   python3 ./udlex/install-dictionaries.py ./config/udlex-224-20261120
   cp ./udlex/90-udlex.options ./config/jvm.options.d/
   ```

   Для отдельного тома `plugins/` или `jvm.options.d/` положите в него соответствующие файлы из нового образа. Если монтируете только отдельные файлы вроде `opensearch.yml`, достаточно содержимого образа.

3. Соберите образ и пересоздайте узел на прежнем томе.

   ```sh
   docker compose build "$svc"
   docker compose up -d --no-deps --pull never "$svc"
   ```

   Во время сборки старый узел обслуживает запросы. На одиночном узле будет перерыв при перезапуске. В кластере обновляйте по одному узлу и дожидайтесь восстановления реплик. Для `docker run` соберите образ командой `docker build -t local/opensearch-udlex:3.10.0-224 ./udlex` и используйте его с прежними томами и параметрами.

4. В Dev Tools выполните `GET /_cat/plugins?v` и проверьте `analysis-udlex` на каждом узле. Запросы из примера возвращают `кот` для «котами» и `cat` для «cats». Для нового индекса возьмите `settings.analysis` из готового файла и задайте анализатор каждому языковому полю. Для существующих документов создайте индекс с новым mapping, перенесите данные через `_reindex` с учётом поступающих записей и переключите алиас. [Готовые запросы](example.http) · [Настройки 224 языков](analysis-224.json)

## Содержимое комплекта

`udlex/` содержит Dockerfile, `plugins/analysis-udlex-3.10.0.zip`, каталог `dictionaries/` со всеми архивами и `DICTIONARIES.json`. Сборка проверяет SHA-256 плагина, каталога и каждого архива. У 39 языковых пакетов есть некоммерческие условия. [Языки и лицензии](LANGUAGES.txt)

При работе из Git возьмите `DICTIONARIES.json` и `dictionaries/` из комплекта. ZIP можно также собрать из исходников и скопировать в `plugins/`. Для собственного ZIP обновите `plugins/SHA256SUMS` командой `(cd plugins && sha256sum analysis-udlex-3.10.0.zip > SHA256SUMS)`. [Сборка плагина](https://github.com/gulldan/udlex/tree/feat/opensearch-3.10/integrations/opensearch/plugin#сборка-из-исходников)
