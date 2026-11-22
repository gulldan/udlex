# udlex

[English](README.md) · [Русский](README.ru.md) · **简体中文**

udlex 是一个基于词典的词形还原工具。它在由 Universal Dependencies 树库和开放词典
学习得到的编译表中查找每个词，并用后缀规则猜测未登录词。核心用 Rust 编写；本包
提供 Python 绑定和命令行，同一核心也可以编译成 C 库。它不使用神经网络：不需要
GPU，结果是确定的。

## 安装

使用 [uv](https://docs.astral.sh/uv/) 把引擎和词典加入项目：

```sh
uv add udlex
uv add "udlex-dict-eng-nc @ https://github.com/gulldan/udlex/releases/download/v1.0.0/udlex_dict_eng_nc-1.0.0-py3-none-any.whl"
uv run udlex lemma eng mice were running
```

需要 Python 3.11 或更高版本。已为 Linux（x86_64、aarch64）、macOS（arm64、
x86_64）和 Windows（x86_64）构建 wheel；在其他平台上 uv 会从源码构建，需要 Rust
工具链。

引擎发布在 PyPI 上，词典不在。每种语言是
[v1.0.0 发布版](https://github.com/gulldan/udlex/releases/tag/v1.0.0)中的一个独立
wheel，用 wheel 地址执行 `uv add` 会像普通依赖一样把它写入 `uv.lock`。
`uv run udlex install --dry-run fin rus` 打印任意语言的地址。
`udlex install <iso> ...` 用 `uv pip`（没有 uv 时用 pip）把它们装进当前环境；在 uv
项目中它会打印对应的 `uv add` 命令，因为 `uv sync` 会删除锁文件中没有的包。

wheel 的文件名是 `udlex_dict_<iso>-1.0.0-py3-none-any.whl`，[非商业](#许可证)包
（例如英语）在代码后加 `_nc`。它们安装到 `udlex_bundle` 命名空间，`udlex.load`
和命令行会自动找到。每个包也以 `udlex-dict-<iso>-1.0.0-bundle.tar.gz`（`-nc`
同理）发布，解压后得到 `bundle/<iso>`。多个归档可以解压到同一位置；没有安装词典
wheel 时 udlex 读取 `./bundle`，也可以用 `UDLEX_BUNDLE` 指定目录，或调用
`udlex.load(directory, iso)`。

## 使用

```py
import udlex

lexicon = udlex.load("eng")  # 每个进程加载一次；可在多线程间共享
print(lexicon.lemmatize(["The", "mice", "were", "running"]))
```

输入是一组词（token）：udlex 不做分词。语言用 ISO 639-3 代码表示。命令行：

```sh
uv run udlex lemma eng mice were running
uv run udlex languages
```

### 无需下载即可试用

下面的例子用一张 `词形<TAB>词元` 表构建一个只有两个词的词典并运行。保存为
`quickstart.py`，然后运行 `uv run quickstart.py`。

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

```sh
uv run udlex --bundle example-bundle lemma -q eng cats mice
```

## 文档

文档为英文：

- [快速入门](docs/quickstart.md)
- [API：Python、命令行、C](docs/api.md)
- [语言与准确率](docs/languages.md)
- [用户词典](docs/user-dictionary.md)
- [构建词典](docs/building-dictionaries.md)

## 语言

v1.0.0 发布版包含 224 种语言的词典，每种语言一个包。[语言表](docs/languages.md)
列出每个包的许可证、评测方式和准确率。

## 准确率

在 Universal Dependencies 2.19 的标准词元上按精确匹配计算
（[`evaluation.json`](dictionaries/catalog/evaluation.json)）。*原样复制* 指把每个
词原样返回。

| 语言数 | 评测方式 | udlex，宏平均 % | 原样复制，% |
| --- | --- | ---: | ---: |
| 103 | 官方测试集 | 89.85 | 58.41 |
| 109 | 二折估计（没有训练集） | 73.87 | 57.54 |
| 12 | 没有标准词元：未测量 | — | — |

与其他词形还原工具在相同测试词上的比较（来自
[`benchmarks/generated/numbers.tex`](benchmarks/generated/numbers.tex)）。每一行
只在两个系统都覆盖的语言上取平均，并去掉了对方模型见过我们测试句子的语言，因此
各行之间不可比较。

| 系统 | 语言数 | 该系统，% | udlex，% |
| --- | ---: | ---: | ---: |
| Stanza | 65 | 95.32 | 93.81 |
| UDPipe 2 | 77 | 94.48 | 93.47 |
| Trankit | 46 | 91.32 | 94.44 |
| UDPipe 1 | 44 | 91.36 | 95.16 |
| simplemma | 33 | 90.75 | 95.44 |
| spaCy sm | 17 | 89.55 | 95.77 |
| spaCy lg | 17 | 90.63 | 95.77 |
| spaCy trf | 7 | 93.98 | 97.14 |
| LemmaGen3 | 16 | 82.27 | 96.33 |
| CoreNLP | 1 | 96.37 | 97.87 |

Stanza 和 UDPipe 2 比 udlex 高约 1 到 1.5 个百分点：它们是能看到整句和词性的神经
网络模型。其余系统平均低于 udlex。

## 速度与内存

所有系统都在同一台机器（AMD Ryzen 9 7950X，16 核，RTX 4090）上，用各系统支持的
全部语言的 UD 2.19 已分词测试句测速。速度是各语言的中位数。“udlex 快多少倍”是
在该系统测过的语言上，udlex Rust 核心单核速度与其之比的几何平均。“并行”取 16 进程
与 16 线程中较好者，神经系统为一块 GPU。“峰值内存”是单核测速进程的常驻内存，包含
测试文本：先是该系统的中位数，再是 udlex 在相同语言上的中位数。

| 系统 | 语言数 | 单核 词/秒 | udlex 快多少倍 | 并行 词/秒 | 峰值内存，MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| **udlex, Rust** | 169 | 54.4M | — | 759M | 147 |
| **udlex, Python** | 169 | 18.0M | — | 142M | 1,172 |
| simplemma | 37 | 6.6M | 5.4 | 57.8M | 324 / 110 |
| LemmaGen3 | 17 | 3.5M | 9.5 | 32.5M | 1,445 / 118 |
| CoreNLP | 1 | 27k | 1,965 | 336k | 1,600 / 111 |
| spaCy sm | 20 | 27k | 1,422 | 218k | 773 / 114 |
| spaCy lg | 20 | 25k | 1,568 | 165k | 1,546 / 114 |
| UDPipe 1 | 50 | 15k | 2,177 | 151k | 153 / 99 |
| Stanza | 90 | 4.5k | 8,160 | 16k (GPU) | 1,240 / 99 |
| spaCy trf | 8 | 278 | 143,271 | 20k (GPU) | 2,840 / 111 |
| Trankit | 53 | 74 | 433,457 | 3.9k (GPU) | 4,665 / 94 |

单核下 udlex 因语言不同每秒处理 620 万到 8.58 亿词。最快的其他系统 simplemma
只在格鲁吉亚语上更快，快 1%。Python 绑定在 16 线程上提速 1.46 倍，在 16 进程上提速
8.9 倍；其内存包含测试文本的 Python 词列表。UDPipe 2 未测速。数字取决于 CPU
和词分布。数值取自 [`numbers.tex`](benchmarks/generated/numbers.tex)，原始
记录见 [`benchmarks/evidence/speed`](benchmarks/evidence/speed)。

## 局限

- **输入是词，输出是词元。** 没有分词器、分句器或词性标注器，请先切分文本。
- **不看上下文。** 每个词只得到一个词元，与句子无关。有多个可能词元的词形（如英语
  *saw*：*see* 或 *saw*）总是得到同一个。这是与 Stanza 和 UDPipe 2 差距的主要来源。
  `lemmatize` 的 `upos=` 和 `context=` 参数只对带上下文键构建的词典起作用，而
  1.0.0 的包构建时没有上下文键。
- **未登录词** 用从树库学到的后缀规则猜测，对词典从未见过的词不如神经网络。
  传入 `use_rules=False`（命令行为 `--lookup-only`）可让这类词原样返回。
- **准确率因语言而异**，从接近 100 % 到不足 50 %；使用前请查看
  [语言表](docs/languages.md)。109 种没有训练集的语言的准确率是小树库上的二折
  估计，另有 12 种语言完全没有标准词元。
- **部分数字带有探索性。** 对没有官方开发集的语言，开发集的替代方案是在第一次
  测试评估之后才引入的。
- **没有语言模型标注的词元。** 发布的词典不包含由语言模型标注的词元；每个包在
  `SOURCES.json` 中列出其来源。
- **非商业树库。** 39 个包（包括英语、捷克语、阿拉伯语和意大利语）学习自采用
  非商业许可证的树库，只能用于非商业用途。

## 其他接口

- **C 库**：`cargo build --locked --release --manifest-path rust/Cargo.toml -p udlex-capi`
  构建 `libudlex_capi`，头文件为 [`rust/capi/include/udlex.h`](rust/capi/include/udlex.h)；
  见 [C API](docs/api.md#c-api)。
- **Go**：[gulldan/udlex-go](https://github.com/gulldan/udlex-go)。

## 许可证

代码采用 [MIT](LICENSE) 许可证。

每个词典包有自己的许可证，由其来源决定：v1.0.0 中有 184 个包采用 CC BY-SA
（182 个为 4.0 版，2 个为 3.0 版），39 个采用 CC BY-NC-SA 4.0，1 个采用 CC BY 4.0。
名称以 `-nc` 结尾的包仅限非商业用途。包内每张表保留其来源的许可证，`SOURCES.json`
像 spaCy 的 `meta.json` 一样列出每个来源的 URL、版本、作者和许可证。分发词典时请
保留其 `NOTICE` 和 `SOURCES.json` 并遵守其许可证。全部来源和包许可证见
[DICTIONARY_LICENSES.md](DICTIONARY_LICENSES.md)。

## 引用

见 [CITATION.cff](CITATION.cff)。参与贡献：[CONTRIBUTING.md](CONTRIBUTING.md)。
