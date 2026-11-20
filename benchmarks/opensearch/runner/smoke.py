"""Verify indexing, matching, phrase queries and highlighting."""

# ruff: noqa: INP001
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import benchmark as bench
import benchmark_threeway as three

root = Path(sys.argv[1])
profile = json.loads((root / "stand/profiles/Udlex.json").read_text())
client = three.make_client(bench, profile["url"], 60)
settings = json.loads(Path(__file__).with_name("index-settings.json").read_text())
settings["settings"]["analysis"]["filter"]["udlex_rus"]["bundle_path"] = "bundle"
name = "udlex-310-russian-smoke"
client.request("PUT", "/" + name, settings)
try:
    tokens, _ = client.request(
        "POST", f"/{name}/_analyze", {"field": "text", "text": "котами"}
    )
    assert [x["token"] for x in tokens["tokens"]] == ["кот"], tokens
    client.request("PUT", f"/{name}/_doc/1?refresh=true", {"text": "Играю с котами"})
    result = {"analysis": tokens}
    for query, value in [("match", "кот"), ("match_phrase", "с котом")]:
        payload = {
            "query": {query: {"text": value}},
            "highlight": {"fields": {"text": {}}},
        }
        response, _ = client.request("POST", f"/{name}/_search", payload)
        hits = response["hits"]["hits"]
        assert len(hits) == 1, response
        assert hits[0]["_id"] == "1", response
        assert any("<em>котами</em>" in s for s in hits[0]["highlight"]["text"]), (
            response
        )
        result[query] = hits[0]
    negative, _ = client.request(
        "POST", f"/{name}/_search", {"query": {"match": {"text": "жираф"}}}
    )
    assert negative["hits"]["total"]["value"] == 0, negative
    result["unrelated_hits"] = 0
    bench.atomic_json(root / "stand/results/Udlex/search-smoke.json", result)
finally:
    client.request("DELETE", "/" + name)
