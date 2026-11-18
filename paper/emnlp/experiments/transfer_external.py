"""Lay out lemma-annotated corpora from outside UD as treebank folders for the transfer screen."""

import argparse
import fnmatch
import html
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_snapshot import read, sha  # noqa: E402

TOKEN = re.compile(r'<w\b[^>]*?lemma="([^"]*)"[^>]*>([^<]*)</w>|<c\b[^>]*>([^<]*)</c>')


def fetch(name, spec, cache):
    path = cache / name / spec["url"].rsplit("/", 1)[-1].split("?")[0]
    if not path.exists():
        path.parent.mkdir(parents=True)
        with (
            urllib.request.urlopen(spec["url"], timeout=300) as remote,
            path.open("wb") as out,
        ):
            shutil.copyfileobj(remote, out)
    if sha(path) != spec["sha256"]:
        raise ValueError(f"{name} differs from its pinned digest")
    return path


def texts(path, file, unpack):
    """Yield (name, text) for one download, one archive member or a member pattern."""
    if unpack == "7z":
        yield (
            path.name,
            subprocess.run(
                ["7z", "e", "-so", path], check=True, capture_output=True
            ).stdout.decode(),
        )
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            wanted = file.get("member") or file["members"]
            for name in sorted(z.namelist()):
                if fnmatch.fnmatchcase(name, wanted):
                    yield name.rsplit("/", 1)[-1], z.read(name).decode()
    else:
        yield path.name, path.read_text(encoding="utf-8")


def rows(text, name):
    """ID FORM LEMMA columns; a first-column 1 or a blank line starts a sentence."""
    doc, sid, tag, sent = None, None, "", []
    for line in text.split("\n"):
        cells = line.rstrip("\r").split("\t")
        if len(cells) >= 3 and cells[0].isdecimal():
            if cells[0] == "1" and sent:
                yield doc or name, sid, tag, sent
                sid, sent = None, []
            sent.append((cells[1], cells[2]))
            continue
        if sent and not line.strip():
            yield doc or name, sid, tag, sent
            sid, sent = None, []
        if line.startswith("# newdoc id = "):
            doc, tag = line.split(" = ", 1)[1].strip(), ""
        elif line.startswith("# sent_id = "):
            sid = line.split(" = ", 1)[1].strip()
        elif line.startswith("# contained_in_datasets = "):
            tag = line.split(" = ", 1)[1]
    if sent:
        yield doc or name, sid, tag, sent


def tei(text, name):
    for found in re.finditer(r'<s xml:id="([^"]+)">(.*?)</s>', text, re.S):
        sent = [
            (html.unescape(w or c), html.unescape(lemma if w or lemma else c))
            for lemma, w, c in TOKEN.findall(found[2])
        ]
        if sent:
            yield ".".join(found[1].split(".")[:3]), found[1], "", sent


def banlemma(text, name):
    sent = []
    for n, line in enumerate(text.split("\n")):
        cells = line.rstrip("\r").split("\t")
        if cells[0] == "<eos>":
            if sent:
                yield name, str(n), "", sent
            sent = []
        elif len(cells) >= 3 and cells[1] and cells[2]:
            sent.append((cells[0], cells[2]))


def people(names):
    """'First Last, First Last' as the 'Last, First; Last, First' of a UD README."""
    found = []
    for name in names.split(","):
        words = name.split()
        if len(words) > 1:
            found.append(words[-1] + ", " + " ".join(words[:-1]))
    return "; ".join(dict.fromkeys(found))


def credits(corpus, specs, cache):
    if "contributors" in corpus:
        return corpus["contributors"]
    if corpus["parser"] == "tei":
        file = corpus["files"][0]
        head = next(
            texts(fetch(file["download"], specs[file["download"]], cache), file, None)
        )[1][:30000]
        return people(",".join(re.findall(r"<name>([^<]*)</name>", head)))
    if "credits" in corpus:
        c = corpus["credits"]
        text = next(texts(fetch(c["download"], specs[c["download"]], cache), c, None))[
            1
        ]
        return people(re.search(rf"^{c['line']}: (.*)$", text, re.M)[1])
    return ""


def main(a):
    plan = read(Path(__file__).with_name("transfer-sources.json"))
    specs = plan["downloads"]
    for corpus in plan["corpora"]:
        parse = {"rows": rows, "tei": tei, "banlemma": banlemma}[corpus["parser"]]
        bank = a.output / corpus["treebank"]
        bank.mkdir(parents=True)
        tokens = 0
        for file in corpus["files"]:
            path = fetch(file["download"], specs[file["download"]], a.cache)
            docs = {}
            for name, text in texts(path, file, corpus.get("unpack")):
                for doc, sid, tag, sent in parse(text, name):
                    docs.setdefault(doc, []).append((sid, tag, sent))
            barred = corpus.get("drop_documents_in")
            with (bank / f"{corpus['config']}-ud-{file['part']}.conllu").open(
                "w"
            ) as out:
                for doc, sents in docs.items():
                    if barred and any(barred in tag for _, tag, _ in sents):
                        continue
                    out.write(f"# newdoc id = {doc}\n")
                    for n, (sid, _, sent) in enumerate(sents, 1):
                        out.write(f"# sent_id = {sid or f'{doc}-{n}'}\n")
                        for i, (form, lemma) in enumerate(sent, 1):
                            cells = [
                                str(i),
                                form.strip() or "_",
                                lemma.strip() or "_",
                            ] + ["_"] * 7
                            out.write("\t".join(cells) + "\n")
                        out.write("\n")
                        tokens += len(sent)
        urls = ", ".join(sorted({specs[f["download"]]["url"] for f in corpus["files"]}))
        (bank / "README.md").write_text(
            f"# {corpus['treebank']}\n\nConverted for the transfer test from {urls}\n\n"
            f"Basis: {corpus['basis']}\nLemmas: {corpus['lemmas']}\n"
            f"Contributors: {credits(corpus, specs, a.cache)}\n"
        )
        print(corpus["config"], tokens, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path, help="one folder per pinned download")
    p.add_argument("output", type=Path)
    main(p.parse_args())
