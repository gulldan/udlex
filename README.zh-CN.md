# udlex

[English](README.md) · [Русский](README.ru.md) · **简体中文**

面向 Python 的快速本地词形还原库。它通过查词典将词形还原为词元，
也可以用后缀规则处理未登录词。输入是一组已经切分好的词。

Go 绑定在独立仓库 [udlex-go](https://github.com/gulldan/udlex-go) 中维护。

## 安装

首次在 PyPI 发布后，使用 `uv` 将 `udlex` 添加到项目。

```sh
uv add udlex
```

在已有的 Python 环境中，可以运行 `python -m pip install udlex`。
需要 Python 3.11 或更高版本。

## 快速开始

用 UTF-8 文本表构建一个小词典。每行包含一个词形和一个词元，以制表符分隔。
将示例保存为 `quickstart.py`，然后运行 `uv run python quickstart.py`。

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

也可以在命令行中使用同一个词典。

```sh
uv run udlex --bundle example-bundle lemma -q eng cats mice
uv run udlex --bundle example-bundle languages
```

如果已有来自可信来源的词典包，将它的目录和 ISO 639-3 语言代码传给 `udlex.load`。
加载一次即可用于后续批量处理。软件包提供词形还原引擎，词典需单独提供。

## 原生库

在 Linux 或 macOS 上使用 Rust 和 Cargo 构建 C 库。该库使用相同的编译词典格式，
运行时无需 Python。

```sh
cargo build --locked --release --manifest-path rust/Cargo.toml -p udlex-capi
```

链接 `rust/target/release` 中的 `libudlex_capi`，并包含头文件
[udlex.h](rust/capi/include/udlex.h)。

代码采用 [MIT 许可证](LICENSE)。
