"""Score language-specific lemmatizers on the census test words, gold tokenization."""

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops it

from verify_snapshot import read, save

COMPARE_DIR = os.environ["UDLEX_COMPARE"]

# Binaries, models and caches of the tools; nothing is written elsewhere.
TOOLS = Path(f"{COMPARE_DIR}/venvs")
LINDAT = "https://lindat.mff.cuni.cz/services/"
CHUNK = 500


def sentences(path):
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            yield [t or "_" for t in row["tokens"]], row["lemmas"]


def align(words, found):
    """Lemmas of a raw-text tool for gold tokens joined by spaces.

    `found` holds (start, end, lemma) of the words the tool saw in that string.
    A token the tool saw nothing in keeps its form; a token the tool cut or
    merged is incomparable (None) and is scored as an error.
    """
    lemmas, at = [], 0
    for word in words:
        end = at + len(word)
        hits = [f for f in found if f[0] < end and f[1] > at]
        if not hits:
            lemmas.append(word)
        elif len(hits) == 1 and hits[0][:2] == (at, end):
            lemmas.append(hits[0][2])
        else:
            lemmas.append(None)
        at = end + 1
    return lemmas


assert align(["a", "b.c", "d"], [(0, 1, "A"), (6, 7, "D")]) == ["A", "b.c", "D"]
assert align(["a", "b.c"], [(2, 3, "B"), (4, 5, "C")]) == ["a", None]
assert align(["a b"], [(0, 1, "A")]) == [None]


def post(url, fields):
    query = urllib.parse.urlencode(fields).encode()
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, query, timeout=600) as reply:  # noqa: S310 - fixed https service
                return json.load(reply)["result"]
        except OSError:
            time.sleep(10 * (attempt + 1))
    raise ConnectionError(f"{url} did not answer")


def spacy_model(name, exclude=()):
    import spacy
    from spacy.tokens import Doc

    nlp = spacy.load(name, exclude=["parser", "ner", *exclude])

    def run(batch):
        docs = (Doc(nlp.vocab, words=s) for s in batch)
        return [[t.lemma_ for t in d] for d in nlp.pipe(docs, batch_size=64)]

    return run, f"{name}-{nlp.meta['version']}"


def latincy(name):
    # The enclitic splitter is the tool's tokenizer; the gold tokens are kept.
    run, model = spacy_model(name, exclude=["enclitic_splitter"])
    return run, model + " without enclitic_splitter"


def classla(code):
    import classla as tool
    import torch

    torch.set_num_threads(6)
    options = {
        "dir": str(TOOLS / "classla-resources"),
        "processors": "tokenize,pos,lemma",
        "logging_level": "ERROR",
    }
    tool.download(code, **options)
    nlp = tool.Pipeline(code, tokenize_pretokenized=True, use_gpu=False, **options)

    def run(batch):
        doc = nlp(batch)
        return [[w.lemma or w.text for w in s.words] for s in doc.sentences]

    return run, f"{code}/standard"


def mystem(_):
    from pymystem3 import Mystem

    tool = Mystem(mystem_bin=str(TOOLS / "mystem" / "mystem"))

    def run(batch):
        answers = []
        for words in batch:
            text, at, found = " ".join(words), 0, []
            items = tool.analyze(text)
            for item in items:
                end = at + len(item["text"])
                if "analysis" in item:
                    readings = item["analysis"]
                    lemma = readings[0]["lex"] if readings else item["text"]
                    found.append((at, end, lemma))
                at = end
            # The offsets hold only while Mystem echoes its input.
            echo = "".join(i["text"] for i in items).rstrip("\n") == text
            answers.append(align(words, found) if echo else None)
        return answers

    return run, "mystem-3.1 -d, first reading, raw text"


def natasha(_):
    from natasha import MorphVocab, NewsEmbedding, NewsMorphTagger

    tagger, vocab = NewsMorphTagger(NewsEmbedding()), MorphVocab()

    def run(batch):
        return [
            [vocab.lemmatize(t.text, t.pos, t.feats) for t in markup.tokens]
            for markup in tagger.map(batch)
        ]

    return run, "slovnet_morph_news_v1 + pymorphy2"


def hanta(model):
    from HanTa.HanoverTagger import HanoverTagger

    tagger = HanoverTagger(model + ".pgz")

    def run(batch):
        return [[lemma for _, lemma, _ in tagger.tag_sent(s)] for s in batch]

    return run, model


def lemminflect(name):
    import lemminflect  # noqa: F401 - registers token._.lemma
    import spacy
    from spacy.tokens import Doc

    nlp = spacy.load(name, exclude=["parser", "ner", "lemmatizer"])

    def run(batch):
        docs = (Doc(nlp.vocab, words=s) for s in batch)
        return [[t._.lemma() for t in d] for d in nlp.pipe(docs, batch_size=256)]

    return run, f"lemminflect on UPOS of {name}-{nlp.meta['version']}"


def morphodita(model):
    def run(batch):
        answers = []
        for start in range(0, len(batch), 200):
            text = "".join("\n".join(s) + "\n\n" for s in batch[start : start + 200])
            fields = {
                "model": model,
                "data": text,
                "input": "vertical",
                "output": "vertical",
                "convert_tagset": "strip_lemma_id",
            }
            rows = [[]]
            for line in post(LINDAT + "morphodita/api/tag", fields).split("\n"):
                if line:
                    rows[-1].append(line.split("\t")[1])
                elif rows[-1]:
                    rows.append([])
            answers.extend(r for r in rows if r)
        return answers

    return run, model + " (LINDAT service, strip_lemma_id)"


def vabamorf(marked):
    from estnltk.vabamorf.morf import Vabamorf

    tool = Vabamorf.instance()

    def lemma(reading):
        # UD Estonian writes "ostu_sedel", "riivi=tud"; Vabamorf keeps these
        # seams only in the root ("sauna_mehe+ks": "+" opens the ending).
        out, rest = "", reading["lemma"]
        for c in reading["root"] if marked else "":
            if c in "_=":
                out += c
            elif rest[:1] == c:
                out, rest = out + c, rest[1:]
            elif c != "+":
                return reading["lemma"]  # the root is not a prefix of the lemma
        return out + rest

    def run(batch):
        return [
            [
                lemma(a["analysis"][0]) if a["analysis"] else word
                for word, a in zip(s, tool.analyze(s), strict=True)
            ]
            for s in batch
        ]

    seams = ", seams _ and = copied from the root" if marked else ""
    return run, "vabamorf disambiguated, first reading" + seams


def camel(name):
    from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
    from camel_tools.disambig.mle import MLEDisambiguator

    kind = BERTUnfactoredDisambiguator if name == "bert" else MLEDisambiguator
    tool = kind.pretrained()

    def run(batch):
        return [
            [
                d.analyses[0].analysis["lex"] if d.analyses else word
                for word, d in zip(s, tool.disambiguate(s), strict=True)
            ]
            for s in batch
        ]

    return run, f"calima-msa-r13 {name} disambiguator, top reading"


def dictabert(name):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(6)
    name, pin = name.split("@")  # the model runs its own code: pin the commit
    tokenizer = AutoTokenizer.from_pretrained(name, revision=pin)
    model = AutoModel.from_pretrained(name, revision=pin, trust_remote_code=True).eval()

    def run(batch):
        answers = []
        for start in range(0, len(batch), 32):
            part = batch[start : start + 32]
            pairs = model.predict([" ".join(s) for s in part], tokenizer)
            for words, row in zip(part, pairs, strict=True):
                # The model returns its own words; find them back in the input.
                text, at, found = " ".join(words), 0, []
                for word, lemma in row:
                    begin = text.find(word, at)
                    if begin < 0:
                        break
                    at = begin + len(word)
                    found.append((begin, at, lemma))
                # A sentence it lost a part of (512 subwords at most) is incomparable.
                whole = len(found) == len(row) and at == len(text)
                answers.append(align(words, found) if whole else None)
        return answers

    return run, f"{name}@{pin[:7]}, raw text"


def latinpipe(model):
    from compare_engines import udpipe_service

    treebank, release = model.rsplit("-", 1)
    return udpipe_service("UD_" + treebank, release)


def morfeusz(_):
    import morfeusz2

    tool = morfeusz2.Morfeusz()

    def lemma(word):
        readings = tool.analyse(word)
        end = max(r[1] for r in readings)
        whole = [r[2][1] for r in readings if r[0] == 0 and r[1] == end]
        # "zamek:Sm3~a" carries a homonym label after the colon.
        return (whole[0].split(":")[0] or whole[0]) if whole else None

    def run(batch):
        return [[lemma(w) for w in words] for words in batch]

    return run, f"{tool.dict_id()}, no disambiguation, first reading, token by token"


def opennlp(code):
    home = TOOLS / "opennlp"
    tool = next(home.glob("apache-opennlp-*/bin/opennlp"))
    pos, lemmas = (
        next(home.glob(f"models/opennlp-{code}-ud-*-{kind}-*.bin"), None)
        for kind in ("pos", "lemmas")
    )
    if not lemmas:
        raise LookupError(f"OpenNLP ships no lemmatizer model for {code}")
    java = "-Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8"
    env = {**os.environ, "JAVA_TOOL_OPTIONS": java}

    def run(batch):
        # The lemmatizer needs tags; the tool's own tagger supplies them.
        done = subprocess.run(  # noqa: S602 - a fixed pipe between two tool commands
            f"{tool} POSTagger {pos} | {tool} LemmatizerME {lemmas}",
            shell=True,
            input="".join(" ".join(s) + "\n" for s in batch),
            capture_output=True,
            encoding="utf-8",
            env=env,
            check=True,
        )
        rows = [[]]
        for line in done.stdout.split("\n"):
            cells = line.split("\t")
            if len(cells) == 3:
                rows[-1].append(cells[2])
            elif rows[-1]:
                rows.append([])
        answers = []
        for words, row in zip(batch, [r for r in rows if r], strict=True):
            # The tool splits on spaces: a token with a space inside is incomparable.
            parts = [len(w.split()) for w in words]
            starts = [sum(parts[:i]) for i in range(len(parts))]
            whole = sum(parts) == len(row)
            answers.append(
                [
                    row[at] if n == 1 else None
                    for at, n in zip(starts, parts, strict=True)
                ]
                if whole
                else None
            )
        return answers

    return run, f"{lemmas.name} on tags of {pos.name}"


# engine: (package or fixed version, loader, {iso: loader argument});
# None for the languages: every paired one, the argument is its two-letter code.
ENGINES = {
    "classla": (
        "classla",
        classla,
        {"slv": "sl", "hrv": "hr", "srp": "sr", "bul": "bg", "mkd": "mk"},
    ),
    "huspacy-lg": ("hu_core_news_lg", spacy_model, {"hun": "hu_core_news_lg"}),
    "huspacy-trf": ("hu_core_news_trf", spacy_model, {"hun": "hu_core_news_trf"}),
    "mystem": ("pymystem3", mystem, {"rus": None}),
    "natasha": ("natasha", natasha, {"rus": None}),
    "hanta": (
        "HanTa",
        hanta,
        {"deu": "morphmodel_ger", "eng": "morphmodel_en", "nld": "morphmodel_dutch"},
    ),
    "lemminflect": ("lemminflect", lemminflect, {"eng": "en_core_web_sm"}),
    "morphodita": (
        "czech-morfflex2.1-pdtc2.0-250909",
        morphodita,
        {"ces": "czech-morfflex2.1-pdtc2.0-250909"},
    ),
    "estnltk": ("estnltk", vabamorf, {"est": False}),
    "estnltk-compound": ("estnltk", vabamorf, {"est": True}),
    "camel-mle": ("camel-tools", camel, {"arb": "mle"}),
    "camel-bert": ("camel-tools", camel, {"arb": "bert"}),
    "dictabert-lex": (
        "transformers",
        dictabert,
        {"heb": "dicta-il/dictabert-lex@91b97a6125732f1c0dda7b479e48050b0fd01ab4"},
    ),
    "latinpipe": (
        "latin-evalatin24-240520",
        latinpipe,
        {"lat": "latin-evalatin24-240520"},
    ),
    "latincy-lg": ("la_core_web_lg", latincy, {"lat": "la_core_web_lg"}),
    "latincy-trf": ("la_core_web_trf", latincy, {"lat": "la_core_web_trf"}),
    "morfeusz2": ("morfeusz2", morfeusz, {"pol": None}),
    "opennlp": ("apache-opennlp-3.0.0", opennlp, None),
}


def score(run, test, keep):
    data = list(sentences(test))
    tokens = correct = lost = lost_tokens = 0
    kept = []
    for start in range(0, len(data), CHUNK):
        part = data[start : start + CHUNK]
        answers = run([words for words, _ in part])
        for (words, gold), answer in zip(part, answers, strict=True):
            lemmas = answer
            if lemmas is None or len(lemmas) != len(words):
                lemmas = [None] * len(words)  # the tool changed the tokenization
            kept.append(lemmas)
            # Incomparable words (None) stay in the count as errors.
            lost += None in lemmas
            for x, y in zip(lemmas, gold, strict=True):
                if y is not None:
                    tokens += 1
                    correct += x == y
                    lost_tokens += x is None
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_bytes(
        gzip.compress(json.dumps(kept, ensure_ascii=False).encode(), mtime=0)
    )
    return {
        "tokens": tokens,
        "correct": correct,
        "sentences": len(data),
        "unaligned_sentences": lost,
        "unaligned_tokens": lost_tokens,
    }


def main(a):
    global TOOLS  # noqa: PLW0603 - the engine loaders read the venv root set here
    TOOLS = a.tools
    inv = read(a.order / "inventory.json")["languages"]
    anchor = a.output / "udpipe2.json"
    anchor = read(anchor)["cells"] if anchor.exists() else {}
    for engine in a.engines:
        package, load, langs = ENGINES[engine]
        out = a.output / f"{engine}.json"
        if out.exists():
            continue
        cells = {}
        paired = {i: r for i, r in inv.items() if r.get("status") == "paired"}
        if langs is None:
            langs = {i: r["config"].split("_")[0] for i, r in sorted(paired.items())}
        for iso, arg in langs.items():
            if iso not in paired:
                continue
            try:
                run, model = load(arg)
            except Exception as error:  # noqa: BLE001 - recorded as unsupported
                reason = f"{type(error).__name__}: {error}"[:300]
                cells[iso] = {"status": "unsupported", "reason": reason}
                print(engine, iso, reason, flush=True)
                continue
            cell = score(
                run,
                a.order / iso / "test.jsonl",
                a.predictions / engine / f"{iso}.json.gz",
            )
            if anchor.get(iso, {}).get("tokens", cell["tokens"]) != cell["tokens"]:
                raise ValueError(f"{engine} was scored on other words for {iso}")
            cells[iso] = {"status": "scored", "model": model, **cell}
            print(
                engine,
                iso,
                cell,
                f"{100 * cell['correct'] / cell['tokens']:.2f}",
                flush=True,
            )
        try:
            release = version(package)
        except PackageNotFoundError:
            release = package
        save(out, {"engine": engine, "version": release, "cells": cells})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("order", "output", "predictions"):
        p.add_argument(key, type=Path)
    p.add_argument("--engines", nargs="+", required=True, choices=sorted(ENGINES))
    p.add_argument("--tools", type=Path, default=TOOLS)
    main(p.parse_args())
