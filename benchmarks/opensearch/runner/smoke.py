"""Verify ten language-specific fields in the same OpenSearch index."""

# ruff: noqa: INP001
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import benchmark as bench
import benchmark_threeway as three

HERE = Path(__file__).resolve().parent
root = Path(sys.argv[1])
profile = json.loads((root / "stand/profiles/Udlex.json").read_bytes())
client = three.make_client(bench, profile["url"], 60)
settings = json.loads((HERE / "index-settings.json").read_bytes())
with (HERE / "search-cases.tsv").open(encoding="utf-8", newline="") as stream:
    cases = list(csv.DictReader(stream, delimiter="\t"))
languages = {case["language"] for case in cases}
fields = settings["mappings"]["properties"]["text"]["properties"]
assert len(cases) == len(languages) == 10
assert languages == fields.keys()

index = "/udlex-demo"
client.request("PUT", index, settings)
try:
    for case in cases:
        language = case["language"]
        client.request(
            "PUT", f"{index}/_doc/{language}", {"text": {language: case["sentence"]}}
        )
    client.request("POST", f"{index}/_refresh")
    count, _ = client.request("GET", f"{index}/_count")
    assert count["count"] == len(cases), count

    checked = {}
    for case in cases:
        language, surface, lemma = case["language"], case["surface"], case["lemma"]
        field = f"text.{language}"
        tokens, _ = client.request(
            "POST", f"{index}/_analyze", {"field": field, "text": surface}
        )
        assert [t["token"] for t in tokens["tokens"]] == [lemma.lower()], (
            field,
            tokens,
        )
        response, _ = client.request(
            "POST",
            f"{index}/_search",
            {"query": {"match": {field: lemma}}, "highlight": {"fields": {field: {}}}},
        )
        hits = response["hits"]["hits"]
        assert [hit["_id"] for hit in hits] == [language], (field, response)
        assert any(f"<em>{surface}</em>" in s for s in hits[0]["highlight"][field]), (
            hits
        )
        checked[language] = {"analysis": tokens, "match": hits[0]}

    response, _ = client.request(
        "POST",
        f"{index}/_search",
        {
            "query": {
                "multi_match": {"query": "котами", "fields": ["text.rus", "text.ukr"]}
            }
        },
    )
    cross_language = sorted(hit["_id"] for hit in response["hits"]["hits"])
    assert cross_language == ["rus", "ukr"], response
    phrase, _ = client.request(
        "POST",
        f"{index}/_search",
        {
            "query": {"match_phrase": {"text.rus": "с котом"}},
            "highlight": {"fields": {"text.rus": {}}},
        },
    )
    hits = phrase["hits"]["hits"]
    assert [hit["_id"] for hit in hits] == ["rus"], phrase
    assert any("<em>котами</em>" in s for s in hits[0]["highlight"]["text.rus"]), hits
    negative, _ = client.request(
        "POST",
        f"{index}/_search",
        {"query": {"multi_match": {"query": "жираф", "fields": ["text.*"]}}},
    )
    assert negative["hits"]["total"]["value"] == 0, negative
    out = root / "stand/results/Udlex"
    bench.atomic_json(
        out / "multilingual-search-smoke.json",
        {
            "index": index[1:],
            "documents": count["count"],
            "languages": checked,
            "cross_language_hits": cross_language,
        },
    )
    bench.atomic_json(
        out / "search-smoke.json",
        {
            **checked["rus"],
            "match_phrase": hits[0],
            "unrelated_hits": 0,
        },
    )
finally:
    client.request("DELETE", index)
