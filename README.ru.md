# udlex

[English](README.md) · **Русский** · [简体中文](README.zh-CN.md)

Быстрая локальная лемматизация для Python. Библиотека ищет словоформы в словаре
и может применять суффиксные правила к незнакомым словам. На вход подаётся
список токенов.

Биндинг для Go находится в репозитории [udlex-go](https://github.com/gulldan/udlex-go).
Фильтр для OpenSearch и пример настройки индекса описаны в [руководстве по интеграции](integrations/opensearch/plugin/README.md).
Сравнение с анализаторами OpenSearch на 224 языках можно повторить в Docker. [Автономный бенчмарк](benchmarks/opensearch/README.md)

## Установка

После первого релиза в PyPI добавьте `udlex` в проект через `uv`.

```sh
uv add udlex
```

Для существующего окружения используйте `python -m pip install udlex`.
Требуется Python 3.11 или новее.

## Быстрый старт

Соберите небольшой словарь из таблицы UTF-8. В каждой строке укажите словоформу
и лемму, разделённые табуляцией. Сохраните пример как `quickstart.py` и запустите
`uv run python quickstart.py`.

```python
from pathlib import Path

import udlex

bundle = Path("example-bundle")
bundle.mkdir()
words = bundle / "lemmas.tsv"
words.write_text("cats\tcat\nmice\tmouse\n", encoding="utf-8")

udlex.build_lexicon(
    str(bundle / "eng"),
    train=None,
    dev=None,
    test=None,
    paradigms=[("example", str(words), False, False)],
    name="eng",
    language="English",
    iso639_3="eng",
)
lexicon = udlex.load(bundle, "eng")
print(lexicon.lemmatize(["cats", "mice", "unknown"]))
# ['cat', 'mouse', 'unknown']
```

Этот же словарь можно использовать из командной строки.

```sh
uv run udlex --bundle example-bundle lemma -q eng cats mice
uv run udlex --bundle example-bundle languages
```

Если у вас уже есть словарь из доверенного источника, передайте в `udlex.load`
путь к набору словарей и код языка ISO 639-3. Загружайте словарь один раз и
используйте его для следующих запросов. Пакет содержит движок. Словарь
подключается отдельно.

## Нативная библиотека

Соберите C-библиотеку на Linux или macOS с помощью Rust и Cargo. Она читает
тот же формат скомпилированных словарей и работает без Python.

```sh
cargo build --locked --release --manifest-path rust/Cargo.toml -p udlex-capi
```

Подключите `libudlex_capi` из `rust/target/release` и заголовок
[udlex.h](rust/capi/include/udlex.h).

Код распространяется по лицензии [MIT](LICENSE).

[Пакеты словарей](https://github.com/gulldan/udlex/releases/tag/v1.0.0) содержат
свои лицензии и сведения об авторах. Если поставляете словарь вместе с
приложением, сохраните эти сведения и соблюдайте условия лицензии словаря.
Пакеты с суффиксом `-nc` предназначены только для некоммерческого использования.
