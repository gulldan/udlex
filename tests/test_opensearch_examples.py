import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations/opensearch"
BENCHMARK = ROOT / "benchmarks/opensearch"
RUNNER = BENCHMARK / "runner"


def _cases() -> list[dict[str, str]]:
    with (RUNNER / "search-cases.tsv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_ten_language_settings_use_the_shipped_analyzers() -> None:
    languages = [case["language"] for case in _cases()]
    assert len(languages) == len(set(languages)) == 10
    lock = json.loads((BENCHMARK / "benchmark.lock.json").read_bytes())
    assert lock["smoke_languages"] == languages

    settings = json.loads((RUNNER / "index-settings.json").read_bytes())
    public_example = INTEGRATION / "plugin/examples/index-settings.json"
    assert json.loads(public_example.read_bytes()) == settings
    full = json.loads((INTEGRATION / "docker/analysis-224.json").read_bytes())
    shipped = full["settings"]["analysis"]
    assert len(shipped["filter"]) == len(shipped["analyzer"]) == 224
    assert {spec["bundle_path"] for spec in shipped["filter"].values()} == {"udlex"}
    fields = settings["mappings"]["properties"]["text"]["properties"]
    assert fields.keys() == set(languages)
    for code in languages:
        analyzer = fields[code]["analyzer"]
        spec = settings["settings"]["analysis"]["analyzer"][analyzer]
        assert spec == shipped["analyzer"][analyzer]
        dictionary = settings["settings"]["analysis"]["filter"][spec["filter"][0]]
        assert dictionary == shipped["filter"][spec["filter"][0]]
        assert dictionary["language"] == code


def test_console_example_indexes_and_searches_every_language() -> None:
    cases = _cases()
    example = (INTEGRATION / "docker/example.http").read_text(encoding="utf-8")
    requests = {}
    for block in example.strip().split("\n\n"):
        request, *body = [
            line for line in block.splitlines() if not line.startswith("#")
        ]
        if request.startswith(("PUT ", "POST ")):
            requests[request] = body
    settings = json.loads((RUNNER / "index-settings.json").read_bytes())
    assert json.loads("\n".join(requests["PUT /udlex-demo"])) == settings
    bulk = list(map(json.loads, requests["POST /udlex-demo/_bulk?refresh=true"]))
    queries = list(map(json.loads, requests["POST /udlex-demo/_msearch"]))
    assert bulk[::2] == [{"index": {"_id": c["language"]}} for c in cases]
    assert bulk[1::2] == [{"text": {c["language"]: c["sentence"]}} for c in cases]
    assert queries[::2] == [{} for _ in cases]
    assert queries[1::2] == [
        {"query": {"match": {f"text.{c['language']}": c["lemma"]}}} for c in cases
    ]
