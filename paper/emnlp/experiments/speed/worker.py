"""One lemmatizer on one language, driven by bench.py: one JSON object per line on stdin/stdout.

Loads the system on the call path compare_engines.py scored it with, reports the load, then
times windows. A window is an independent sample of test sentences drawn with replacement;
its answers are checked against the predictions the quality was scored from.
Runs in each system's own environment (Python 3.10 for Trankit), started with ``python -I``.
"""

import gzip
import json
import os
import random
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path

PROTOCOL = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)  # libraries that print to stdout would break the protocol
OPENED = []
ARMED = [False]
CODE = (".py", ".pyc", ".pyi", ".so", ".pth")


def audit(event, args):
    if ARMED[0] and event == "open" and isinstance(args[0], (str, bytes)):
        OPENED.append(os.fsdecode(args[0]))


sys.addaudithook(audit)


def send(obj):
    PROTOCOL.write(json.dumps(obj, ensure_ascii=False) + "\n")


def memory(pid="self"):
    """VmHWM and VmRSS of a process in KiB."""
    rows = Path(f"/proc/{pid}/status").read_text().splitlines()
    return {
        k.lower() + "_kb": int(v.split()[0])
        for k, v in (r.split(":", 1) for r in rows)
        if k in ("VmHWM", "VmRSS")
    }


def data_files(paths):
    """Model files a load read, without code and system files."""
    keep = set()
    for p in paths:
        if p.startswith(("/proc", "/sys", "/dev", "/etc", "/usr")):
            continue
        if p.endswith(CODE) or ".so." in p or ".dist-info" in p or "__pycache__" in p:
            continue
        if os.path.isfile(p):
            keep.add(os.path.abspath(p))
    return sorted(keep)


class Plain:
    """``run(batch)`` on token lists: the call shape of compare_engines.cell."""

    def __init__(self, forms, run, reset=None):
        self.forms, self.run, self.reset = forms, run, reset or (lambda: None)
        self.size = None  # sentences per call; None passes the whole window at once

    def prepare(self, picks):
        return [self.forms[i] for i in picks]

    def call(self, payload):
        if not self.size:
            return self.run(payload)
        step = self.size
        return [
            r
            for i in range(0, len(payload), step)
            for r in self.run(payload[i : i + step])
        ]

    def extract(self, picks, raw, stride):
        return [(picks[k], raw[k]) for k in range(0, len(picks), stride)]


class UdlexPy(Plain):
    """``Lexicon.lemmatize(tokens, sentence_initial=mask)`` on flat batches of sentences."""

    def __init__(self, forms, lexicon, batch, threads):
        super().__init__(forms, None)
        self.lexicon, self.batch = lexicon, batch
        self.pool = ThreadPoolExecutor(threads) if threads > 1 else None
        self.masks = [[i == 0 for i in range(len(s))] for s in forms]

    def prepare(self, picks):
        chunks = []
        for start in range(0, len(picks), self.batch):
            part = picks[start : start + self.batch]
            flat = list(chain.from_iterable(self.forms[i] for i in part))
            mask = list(chain.from_iterable(self.masks[i] for i in part))
            chunks.append((flat, mask))
        return chunks

    def call(self, chunks):
        f = self.lexicon.lemmatize
        if self.pool:
            return list(self.pool.map(lambda c: f(c[0], sentence_initial=c[1]), chunks))
        return [f(flat, sentence_initial=mask) for flat, mask in chunks]

    def extract(self, picks, raw, stride):
        out, k = [], 0
        for chunk in raw:
            offset = 0
            for _ in range(min(self.batch, len(picks) - k)):
                n = len(self.forms[picks[k]])
                if k % stride == 0:
                    out.append((picks[k], chunk[offset : offset + n]))
                offset, k = offset + n, k + 1
        return out


def corenlp(cfg, forms, threads):
    """CoreNLP 4.6.0 as a server with the CLI properties of compare_engines.corenlp_pipeline."""
    import socket
    import subprocess
    import urllib.error
    import urllib.parse
    import urllib.request

    home = Path(cfg["models"])
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    java = subprocess.Popen(
        [
            "java",
            "-Xmx6g",
            "-cp",
            f"{home}/*",
            "edu.stanford.nlp.pipeline.StanfordCoreNLPServer",
            *("-port", str(port), "-threads", str(threads)),
            *("-maxCharLength", "2000000000", "-timeout", "3600000", "-quiet", "true"),
        ],
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    props = {
        "annotators": "tokenize,pos,lemma",
        "tokenize.whitespace": "true",
        "ssplit.eolonly": "true",
        "outputFormat": "conll",
        "output.columns": "word,lemma",
    }
    url = f"http://127.0.0.1:{port}/?properties=" + urllib.parse.quote(
        json.dumps(props)
    )

    def post(batch):
        text = "".join(" ".join(t or "_" for t in s) + "\n" for s in batch)
        request = urllib.request.Request(
            url,
            data=text.encode(),
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )
        with urllib.request.urlopen(request, timeout=3600) as reply:
            blocks = reply.read().decode("utf-8")
        return [
            [row.split("\t")[1] for row in block.split("\n")]
            for block in blocks.strip("\n").split("\n\n")
        ]

    while True:  # the server answers once it listens
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            if java.poll() is not None:
                raise RuntimeError("CoreNLP server exited") from None
            time.sleep(0.05)
    pool = ThreadPoolExecutor(threads) if threads > 1 else None

    def run(batch):
        if not pool:
            return post(batch)
        size = -(-len(batch) // threads)
        parts = [batch[i : i + size] for i in range(0, len(batch), size)]
        return [row for part in pool.map(post, parts) for row in part]

    jar = next(home.glob("stanford-corenlp-*-models.jar"))
    entry = "edu/stanford/nlp/models/pos-tagger/english-left3words-distsim.tagger"
    disk = zipfile.ZipFile(jar).getinfo(entry).file_size
    return Plain(forms, run), java, [f"{jar}!{entry}"], disk


def build(cfg, forms):
    """The system and what its load cost: (system, model, files, import_s, load_s, extras)."""
    name, lang, threads, gpu = cfg["system"], cfg["lang"], cfg["threads"], cfg["gpu"]
    sys.path.insert(0, cfg["experiments"])
    files, extras = [], {}
    t = time.perf_counter()
    if name in ("stanza", "trankit", "spacy-trf"):
        import torch

        torch.set_num_threads(threads)
    if name.startswith("spacy") and gpu:
        sys.path.insert(0, cfg["cupy"])
    if name == "trankit" or name.startswith("spacy"):
        import compare_engines
    imported = time.perf_counter()
    if name == "udlex-py":
        import udlex

        imported = time.perf_counter()
        lexicon = udlex.load(cfg["bundle"], cfg["iso"], use_user=False)
        system = UdlexPy(forms, lexicon, cfg["batch"], threads)
        model = f"udlex {udlex.__version__} {cfg['iso']}"
        files = [cfg["bundle"]]
    elif name == "stanza":
        import stanza

        imported = time.perf_counter()
        code, package = cfg["model"].split("/")
        # compare_engines.stanza_pipeline without its download check; same arguments.
        nlp = stanza.Pipeline(
            code,
            dir=cfg["models"],
            package=package,
            processors="tokenize,pos,lemma",
            tokenize_pretokenized=True,
            use_gpu=gpu,
            download_method=None,
            logging_level="ERROR",
            **cfg.get("stanza_kwargs", {}),
        )

        def run(batch):
            doc = nlp([[t or "_" for t in s] for s in batch])
            return [
                [w.lemma if w.lemma is not None else w.text for w in s.words]
                for s in doc.sentences
            ]

        system, model = Plain(forms, run), cfg["model"]
    elif name == "trankit":
        import trankit

        imported = time.perf_counter()
        if gpu:
            plain = trankit.Pipeline
            trankit.Pipeline = lambda *a, **k: plain(*a, **dict(k, gpu=True))
        run, model = compare_engines.trankit_pipeline(
            lang, cfg["treebank"], Path(cfg["models"])
        )
        torch.set_num_threads(threads)  # trankit_pipeline sets 6
        system = Plain(forms, run)
        snapshot = Path(cfg["models"], "xlm-roberta-large", "models--xlm-roberta-large")
        files = [
            str(snapshot),
            str(Path(cfg["models"], "xlm-roberta-large", model.split("/")[0])),
        ]
    elif name.startswith("spacy"):
        import spacy

        imported = time.perf_counter()
        if gpu:
            spacy.require_gpu()
        size = compare_engines.SPACY_SIZE[name]
        run, model = compare_engines.spacy_pipeline(lang, size)
        if size == "trf":
            torch.set_num_threads(threads)  # spacy_pipeline sets 6
        system = Plain(forms, run)
        files = [str(spacy.util.get_package_path(model.rsplit("-", 1)[0]))]
    elif name in ("simplemma", "lemmagen3"):
        sys.path.insert(0, cfg["repo"])
        from bench import adapters

        imported = time.perf_counter()
        adapters.configure_project_root(cfg["repo"])
        adapter = adapters.create_adapter(name)
        adapter.load(lang, cfg["config"])
        system = Plain(forms, adapter.run, adapter.reset)
        model = str(getattr(adapter, "model", None) or name)
        if name == "lemmagen3":
            import lemmagen3

            # The bench adapter memoises every form; that is harness, not LemmaGen3,
            # and on a sample of one test set nearly every call would hit it.
            lem = adapter.l.lemmatize
            system = Plain(
                forms, lambda sents: [[lem(t) or t for t in s] for s in sents]
            )
            files = [str(Path(lemmagen3.__file__).parent / "models" / f"{lang}.bin")]
    elif name == "udpipe1":
        from ufal.udpipe import Model, Pipeline, ProcessingError

        imported = time.perf_counter()
        tagger = Model.load(cfg["model_file"])
        if tagger is None:
            raise RuntimeError(f"cannot load {cfg['model_file']}")
        pipe = Pipeline(tagger, "conllu", Pipeline.DEFAULT, Pipeline.NONE, "conllu")

        def run(batch):
            # What compare_engines.udpipe_service sends to LINDAT and reads back.
            text = "".join(
                "".join(
                    f"{i}\t{t or '_'}" + "\t_" * 8 + "\n" for i, t in enumerate(s, 1)
                )
                + "\n"
                for s in batch
            )
            error = ProcessingError()
            result = pipe.process(text, error)
            if error.occurred():
                raise RuntimeError(error.message)
            rows = [[]]
            for line in result.split("\n"):
                cells = line.split("\t")
                if len(cells) == 10 and cells[0].isdecimal():
                    rows[-1].append(cells[2])
                elif not line and rows[-1]:
                    rows.append([])
            return [r for r in rows if r]

        system, model, files = Plain(forms, run), cfg["model"], [cfg["model_file"]]
        system.tagger = tagger  # the SWIG pipeline does not own its model
    elif name == "corenlp":
        system, java, files, disk = corenlp(cfg, forms, threads)
        model, extras = (
            "corenlp 4.6.0 english-left3words-distsim",
            {"java": java, "disk_bytes": disk},
        )
    else:
        raise ValueError(f"unknown system {name}")
    loaded = time.perf_counter()
    system.size = cfg.get("call_sentences")
    return system, model, files, imported - t, loaded - imported, extras


def references(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    cfg = json.loads(sys.argv[1])
    if cfg["system"] == "trankit":  # torch 2.0.1 rejects the expandable_segments option
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    with open(cfg["test"], encoding="utf-8") as fh:
        forms = [json.loads(line)["tokens"] for line in fh]
    expected = references(cfg["ref"])
    if len(expected) != len(forms):
        raise ValueError("reference and test differ in length")
    lengths = [len(s) for s in forms]
    ARMED[0] = True
    try:
        system, model, files, import_s, load_s, extras = build(cfg, forms)
        t = time.perf_counter()
        system.call(system.prepare([0]))
        first_s = time.perf_counter() - t
    except Exception as error:  # noqa: BLE001 - reported to the driver as a failed cell
        send({"event": "error", "reason": f"{type(error).__name__}: {error}"[:400]})
        return
    ARMED[0] = False
    java = extras.get("java")
    send(
        {
            "event": "ready",
            "model": model,
            "import_s": import_s,
            "load_s": load_s,
            "first_s": first_s,
            "files": data_files(OPENED) + files,
            "disk_bytes": extras.get("disk_bytes"),
            **{"rss_load_kb": memory(java.pid if java else "self")["vmrss_kb"]},
        }
    )
    state = {"picks": [], "payload": None}
    for line in sys.stdin:
        try:
            if serve(json.loads(line), state, system, forms, expected, lengths, java):
                return
        except Exception as error:  # noqa: BLE001 - CUDA OOM in a window, say: reported
            send({"event": "error", "reason": f"{type(error).__name__}: {error}"[:400]})
            return


def serve(command, state, system, forms, expected, lengths, java):  # noqa: PLR0913
    """Answer one driver command; True once the worker should exit."""
    what = command["cmd"]
    if what == "warm":
        # Grow the sample until a window lasts as long as the timed ones will.
        goal, floor = command.get("min_window", 0.0), command.get("min_sentences", 1)
        n, spent, rounds, dt, last = 1, 0.0, 0, 0.0, 0
        while spent < command["seconds"] or rounds < 2 or dt < goal or last < floor:
            draw = random.Random(10**9 + rounds).choices(range(len(forms)), k=n)
            system.reset()
            ready = system.prepare(draw)
            t = time.perf_counter()
            system.call(ready)
            dt = time.perf_counter() - t
            rate = sum(lengths[i] for i in draw) / dt
            spent, rounds, last = spent + dt, rounds + 1, n
            if dt < goal:
                n = max(n + 1, min(16 * n, int(1.2 * n * goal / max(dt, 1e-4))))
            n = max(n, floor)
        send({"rate": rate, "windows": rounds})
    elif what == "prep":
        picks = random.Random(command["seed"]).choices(
            range(len(forms)), k=command["sentences"]
        )
        system.reset()  # memoising adapters start every window cold, as bench intends
        state["picks"], state["payload"] = picks, system.prepare(picks)
        send({"tokens": sum(lengths[i] for i in picks)})
    elif what == "go":
        picks = state["picks"]
        cpu = sum(os.times()[:2])
        t = time.perf_counter()
        raw = system.call(state["payload"])
        seconds = time.perf_counter() - t
        cpu = sum(os.times()[:2]) - cpu
        state["payload"] = None
        checked = mismatch = 0
        bad = []
        for i, lemmas in system.extract(picks, raw, max(1, len(picks) // 2000)):
            checked += len(expected[i])
            wrong = sum(a != b for a, b in zip(lemmas, expected[i], strict=False))
            wrong += abs(len(lemmas) - len(expected[i]))
            mismatch += wrong
            if wrong and len(bad) < 3:
                bad.append([i, lemmas[:40], expected[i][:40]])
        send(
            {
                "tokens": sum(lengths[i] for i in picks),
                "seconds": seconds,
                "cpu_s": cpu,
                "checked": checked,
                "mismatch": mismatch,
                "bad": bad,
            }
        )
    elif what == "exit":
        out = memory()
        if java:
            out["java"] = memory(java.pid)
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            cuda = sys.modules["torch"].cuda
            out["cuda_reserved_kb"] = cuda.max_memory_reserved() // 1024
            out["cuda_allocated_kb"] = cuda.max_memory_allocated() // 1024
        send(out)
        if java:
            java.terminate()
            java.wait(30)
        return True
    return False


if __name__ == "__main__":
    main()
