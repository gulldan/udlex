# Словари для UDlex 1.0.0 на данных UD 2.19

Исследовательская сборка содержит 172 отдельных Python wheels и 172 переносимых архива. Версия движка — `1.0.0`, версия данных — `1.0.0.post1`. Это материалы исследовательской ветки. Контрольные суммы, точная ревизия каждого источника, обучающие строки, число правил и условия использования записаны для каждого языка. [Каталог](DICTIONARIES.json) · [Пакеты](packages) · [Проверка артефактов](verification.json) · [Проверка установки](installed.json)

## Из чего обучены словари

| Вход | Языков | Как его использовали |
| --- | ---: | --- |
| Корпус UD 2.19 | 100 | Взяли заявленный для пакета корпус. Использовали train, при его отсутствии dev, затем test. В тесте качества исключили участвовавшие в сборке документы. |
| Туркменский UD_Turkmen-TUD | 1 | Корпус отсутствует в 2.19. Заново обучили по ранее закреплённому исходному файлу с проверкой его SHA-256. |
| UniMorph, CC-CEDICT и OpenCorpora | 55 | Заново извлекли пары из закреплённых версий источников. Для UniMorph проверили каждую часть парадигм. |
| Внешние таблицы «форма — лемма» | 16 | Воспроизвели точные входные таблицы, для каждой сверили байты с хешем исходной проекции. Они включены в каталог, чтобы чистая сборка могла читать TSV напрямую. |

Для 16 источников исходный снимок распространялся в виде сжатой таблицы точного внешнего поиска. При подготовке извлекали только эту таблицу из контейнера, который не содержал обученных правил или таблиц корпусного поиска. В финальном прогоне все 172 сборки читали исходные наблюдения или проверенные TSV. Для словарных ресурсов правила учились на словоформах. Для корпусов сохранены частоты наблюдений и позиции слов в предложениях. У испанского жестового языка (`ssp`) источник содержит идентификаторы жестов. Для него собрана таблица тождественных соответствий. У остальных 171 языков есть обученные суффиксные правила. [Входные таблицы и их коды ISO](source-tables) · [Дистрибутив UD 2.19](https://universaldependencies.org/download.html)

Для `lzh`, `naq` и `qtd` декларация в README корпуса расходится с приложенным файлом LICENSE. Обе версии сохранены в пакетах и отмечены полем `requires_license_review`. Атрибуция и дополнительные лицензии внешних таблиц находятся в соответствующем wheel в каталоге `.dist-info/licenses/` и в переносимом архиве в `LICENSES/`.

## Установка

В окружении с движком 1.0.0 установите wheel нужного языка. Имена всех дистрибутивов, в том числе вариантов с суффиксом `-nc`, приведены в каталоге.

~~~sh
python -m pip install --no-index --no-deps dictionaries/ud-2.19/packages/udlex_dict_eng-1.0.0.post1-py3-none-any.whl
python -c 'import udlex; print(udlex.load("eng").lemmatize(["dogs"]))'
~~~

## Воспроизведение чистой сборки

Команды рассчитаны на Linux. Запускайте их из корня исследовательской ветки. Переменная `assets` указывает на каталог исходного черновика с файлом DICTIONARIES.json и wheels. Их можно получить через `paper/emnlp/experiments/fetch_assets.py`. В примере `python` запускает Python с установленным каноническим движком 1.0.0. Для чтения исходного OpenCorpora нужен `pymorphy3==2.0.6`. Сборка проверяет SHA-256 официального архива и байты распакованных файлов.

~~~sh
run_dir=$(mktemp -d)
data_dir="$PWD/dictionaries/ud-2.19"
mkdir -p "$run_dir/native" "$run_dir/ud"
git archive 7afe2fe8f3ea6ea56fff835bbc61d8af49da731f rust rust-toolchain.toml | tar -x -C "$run_dir/native"
mkdir -p "$run_dir/native/rust/src/bin"
cp "$data_dir/dictionary_io.rs" "$run_dir/native/rust/src/bin/"
(cd "$run_dir/native" && cargo build --locked --manifest-path rust/Cargo.toml --profile dev-fast --no-default-features --features train --bin dictionary_io)
binary="$run_dir/native/rust/target/dev-fast/dictionary_io"
curl -fL https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/bd57416d-f1ec-43a0-8ebd-0a53cb9bda59/content -o "$run_dir/ud219.tgz"
tar -xzf "$run_dir/ud219.tgz" -C "$run_dir/ud"
python "$data_dir/fetch_sources.py" "$assets/DICTIONARIES.json" "$run_dir/sources"
python "$data_dir/refresh.py" "$assets" "$run_dir/no-base-models" "$run_dir/ud219.tgz" "$run_dir/ud/ud-treebanks-v2.19" "$run_dir/sources" "$binary" "$run_dir/build" "$run_dir/catalog" --projections "$data_dir/source-tables" --jobs 5
python "$data_dir/verify.py" "$run_dir/catalog" "$run_dir/catalog/verification.json"
~~~

Аргумент `no-base-models` может указывать на несуществующий путь. При наличии 16 проверенных таблиц сборке достаточно исходных данных. В wheels из исходного черновика она обращается к файлам лицензий и атрибуции. Воспроизводимость словаря проверяется по SHA-256 обучающей TSV и идентификатору скомпилированного поколения. Поле `recipe_sha256` учитывает также байты сборочного исполняемого файла, поэтому может зависеть от версии компилятора и состава служебных точек входа.
