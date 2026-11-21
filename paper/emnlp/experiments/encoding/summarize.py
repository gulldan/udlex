#!/usr/bin/env python3
"""Folds a receipts directory of the encoding lab into SUMMARY.md (stdout).

Usage: summarize.py RECEIPTS_DIR
Reads prepare/*.json, census/*.json, bench.jsonl and environment.txt.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(sys.argv[1])
SPLITS = [4, 6, 8, 10, 12, 16]
# The stream the conclusions rest on: independent draws from the test tokens.
MAIN = "iid"


def gm(values):
    values = [v for v in values if v and v > 0]
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def clock(got):
    """(median, minimum) of a run's samples, in thread CPU time where the lab recorded it."""
    return (
        got.get("cpu_ns_median") or got["ns_median"],
        got.get("cpu_ns_min") or got["ns_min"],
    )


def settle(values, noisy):
    """One number from a measurement's runs.

    A quiet machine gets the median over rounds of each run's median. On a shared machine
    contention only ever adds time, so the minimum over every sample is the better estimate.
    """
    return (
        min(v[1] for v in values) if noisy else statistics.median(v[0] for v in values)
    )


def is_single(cand):
    """One structure for the whole table, mapped from a file."""
    plain = not cand.startswith(("split", "bf8>", "call>", "core", "hashbrown"))
    return plain and cand.endswith(("+ids", "+inline"))


def fmt(value, digits=1):
    return "—" if value is None else f"{value:.{digits}f}"


def table(head, rows):
    out = [
        "| " + " | ".join(head) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(head))) + "|",
    ]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


prepared = {
    p.stem: json.loads(p.read_text())["keysets"]
    for p in sorted((OUT / "prepare").glob("*.json"))
}
census = {
    p.stem: json.loads(p.read_text())["keysets"]
    for p in sorted((OUT / "census").glob("*.json"))
}

# (lang, keyset, cand, workload, mode) -> ns: the median over rounds of each run's median sample.
# Rounds left out of the time estimates (`SKIP_ROUNDS=1`), e.g. a round a
# documented disturbance degraded; counters and cold starts are kept.
SKIP = {r for r in os.environ.get("SKIP_ROUNDS", "").split(",") if r}
runs = defaultdict(list)
counters = {}
opens = defaultdict(list)
colds = {}
notes = set()
bench = OUT / "bench.jsonl"
for line in bench.read_text().splitlines() if bench.exists() else []:
    rec = json.loads(line)
    notes.add(rec.get("note"))
    base = (rec["lang"], rec["keyset"], rec["cand"])
    skipped = str(rec.get("round")) in SKIP
    if not skipped:
        opens[base].append(rec["open_warm_ns"])
    if rec.get("cold"):
        colds[base] = rec["cold"]
    for workload, modes in rec["results"].items():
        for mode, got in modes.items():
            if not skipped:
                runs[base + (workload, mode)].append(clock(got))
            if "per_query" in got:
                counters[base + (workload,)] = got["per_query"]
NOISY = "noisy" in notes
ns = {key: settle(values, NOISY) for key, values in runs.items()}
langs = sorted({k[0] for k in ns})
keysets = [k for k in ("exact", "external") if any(key[1] == k for key in ns)]


def t(lang, keyset, cand, workload=MAIN, mode="thr"):
    return ns.get((lang, keyset, cand, workload, mode))


def cands_of(keyset):
    seen = []
    for key in ns:
        if key[1] == keyset and key[2] not in seen:
            seen.append(key[2])
    return seen


def info(lang, keyset, cand):
    return prepared.get(lang, {}).get(keyset, {}).get("candidates", {}).get(cand, {})


def across(keyset, cand, workload=MAIN, mode="thr"):
    return gm([t(lang, keyset, cand, workload, mode) for lang in langs])


def bytes_per_key(keyset, cand):
    values = [info(lang, keyset, cand).get("bytes_per_key") for lang in langs]
    values = [v for v in values if v]
    return statistics.median(values) if values else None


print("# Лаборатория кодирования словаря «форма → лемма»: сводка\n")
if "noisy" in notes:
    print(
        "**Времена сняты на занятой машине (метка `noisy`): годятся для скрининга, не для статьи.**"
    )
    print("Счётчики инструкций и размеры от загрузки почти не зависят.\n")
env = OUT / "environment.txt"
if env.exists():
    print("```text\n" + env.read_text().strip() + "\n```\n")
print(
    "Все времена — нс процессорного времени потока на запрос (на тихом пиннутом ядре равно стенному):"
)
print(
    "минимум по всем сэмплам всех раундов, раз машина занята (соседи только добавляют время);"
    if NOISY
    else "медиана по раундам медианы сэмплов одного процесса;"
)
print(
    "в сводных столбцах — геометрическое среднее по языкам. «Поток» — независимые выборки из токенов теста"
)
print(
    "(их закон Ципфа и естественная доля промахов; см. раздел 12а, почему не дословный повтор). `thr` — независимые запросы,"
)
print("`lat` — зависимая цепочка. Байты на ключ — индекс плюс слой значений.\n")

print("## 0. Стабильность и загрузка машины\n")
if SKIP:
    print(
        f"**В оценки времени не входят раунды {', '.join(sorted(SKIP))}** (счётчики и холодный старт из них сохранены); причина — ниже.\n"
    )
if (OUT / "notes.md").exists():
    print((OUT / "notes.md").read_text().strip() + "\n")
spread = defaultdict(list)
for key, values in runs.items():
    medians = [v[0] for v in values]
    if len(medians) >= 2:
        middle = statistics.median(medians)
        spread[(key[1], key[3], key[4])].append((max(medians) - min(medians)) / middle)
rows = []
for (keyset, workload, mode), values in sorted(spread.items()):
    if workload not in (MAIN, "iid_hit", "iid_miss", "real", "uni_hit"):
        continue
    values.sort()
    rows.append(
        [
            keyset,
            workload,
            mode,
            len(values),
            fmt(100 * statistics.median(values)),
            fmt(100 * values[int(0.9 * (len(values) - 1))]),
            fmt(100 * sum(v > 0.05 for v in values) / len(values)),
        ]
    )
if rows:
    print(
        "Разброс между раундами: (max − min) / медиана по медианам раундов одной пары «язык × кандидат».\n"
    )
    print(
        table(
            [
                "таблица",
                "поток",
                "режим",
                "пар",
                "медиана, %",
                "90-й перцентиль, %",
                "доля пар > 5 %, %",
            ],
            rows,
        )
    )
loads = OUT / "load.jsonl"
if loads.exists():
    samples = [json.loads(line) for line in loads.read_text().splitlines()]
    if samples:
        one = sorted(float(s["loadavg"][0]) for s in samples)
        own = str(samples[0]["cpu"])
        mine = [s["busy_measuring"][own] for s in samples]
        twin = [v for s in samples for k, v in s["busy_measuring"].items() if k != own]
        who = defaultdict(list)
        for sample in samples:
            for proc in sample["top"]:
                who[proc["name"]].append(proc["cpu_pct"])
        print(
            f"Загрузка по минутам ({len(samples)} замеров, {samples[0]['utc']} … {samples[-1]['utc']}): "
            f"load1 мин {one[0]:.2f}, медиана {statistics.median(one):.2f}, макс {one[-1]:.2f}; "
            f"измерительное ядро {own} занято в среднем на {statistics.mean(mine):.0f} %, его SMT-сосед — на "
            f"{statistics.mean(twin) if twin else 0:.1f} % (макс {max(twin) if twin else 0:.1f}); "
            f"вся машина — на {statistics.mean(s['busy_all_mean'] for s in samples):.1f} %.\n"
        )
        rows = [
            [name, len(v), fmt(statistics.mean(v)), fmt(max(v))]
            for name, v in sorted(who.items(), key=lambda x: -sum(x[1]))[:8]
        ]
        print(
            table(["процесс", "минут в топ-5", "CPU, % в среднем", "CPU, % макс"], rows)
        )

print("## 1. Языки и множества ключей\n")
rows = []
for lang, sets in prepared.items():
    for keyset, body in sets.items():
        s = body["stats"]
        rows.append(
            [
                f"{lang}/{keyset}",
                s["keys"],
                fmt(s["key_len"]["mean_bytes"]),
                fmt(s["bytes_per_char"], 2),
                fmt(100 * s["key_len"]["share"]["le8"]),
                fmt(100 * s["stream"]["query_len"]["share"]["le8"]),
                fmt(100 * s["stream"]["hit_rate"]),
                fmt(100 * s["identity_keys"]),
                fmt(100 * s["stream"]["identity_hits"]),
                s["lemmas"],
                s["rules"],
            ]
        )
print(
    table(
        [
            "язык/таблица",
            "ключей",
            "ср. длина, байт",
            "байт/символ",
            "ключей ≤8 Б, %",
            "запросов ≤8 Б, %",
            "попаданий, %",
            "лемма=форма (ключи), %",
            "лемма=форма (попадания), %",
            "лемм",
            "правил",
        ],
        rows,
    )
)

print("## 2. Проверка корректности\n")
bad, approx, total = [], [], 0
for lang, sets in {**census, **prepared}.items():
    for keyset, body in sets.items():
        for cand, entry in body["candidates"].items():
            total += 1
            if not entry["verified"]:
                bad.append(f"{lang}/{keyset}/{cand}: {entry.get('error')}")
            if entry.get("false_positives") or entry.get("caught_panics"):
                approx.append(
                    f"{lang}/{keyset}/{cand}: ложных попаданий {entry['false_positives']}, "
                    f"пойманных паник {entry['caught_panics']}"
                )
print(
    f"Сборок кандидатов проверено против эталона (все ключи и все потоки запросов, побайтно): {total}; "
    f"отказов: {len(bad)}.\n"
)
for line in bad[:40]:
    print(f"- ОТКАЗ {line}")
if approx:
    print("\nНеточные или паникующие на чужих ключах (не отказ, но оговорка):\n")
    for line in approx[:60]:
        print(f"- {line}")
print()

for keyset in keysets:
    names = cands_of(keyset)
    print(f"## 3. Все кандидаты, таблица `{keyset}`\n")
    rows = []
    for cand in names:
        heap = next(
            (info(l, keyset, cand).get("heap") for l in langs if info(l, keyset, cand)),
            "",
        )
        exact = all(info(l, keyset, cand).get("exact", True) for l in langs)
        build = gm(
            [
                info(l, keyset, cand).get("build_ns", 0)
                / max(1, prepared[l][keyset]["stats"]["keys"])
                for l in langs
                if keyset in prepared.get(l, {})
            ]
        )
        rows.append(
            [
                f"`{cand}`",
                fmt(across(keyset, cand)),
                fmt(across(keyset, cand, mode="lat")),
                fmt(across(keyset, cand, "iid_hit")),
                fmt(across(keyset, cand, "iid_hit", "lat")),
                fmt(across(keyset, cand, "iid_miss")),
                fmt(across(keyset, cand, "iid_miss", "lat")),
                fmt(across(keyset, cand, "uni_hit")),
                fmt(across(keyset, cand, "uni_hit", "lat")),
                fmt(across(keyset, cand, "uni_miss")),
                fmt(across(keyset, cand, "cascade")),
                fmt(bytes_per_key(keyset, cand)),
                fmt(build, 0),
                ("" if exact else "неточный ") + (f"куча: {heap}" if heap else ""),
            ]
        )
    rows.sort(key=lambda r: float(r[1]) if r[1] != "—" else 1e9)
    print(
        table(
            [
                "кандидат",
                "поток thr",
                "поток lat",
                "попадания thr",
                "попадания lat",
                "промахи thr",
                "промахи lat",
                "uniform thr",
                "uniform lat",
                "uniform-промахи thr",
                "каскад thr",
                "Б/ключ",
                "сборка, нс/ключ",
                "оговорки",
            ],
            rows,
        )
    )
    print(
        "«Каскад» — пробы, которые до словарной таблицы доходят после корпусной (только для `external`).\n"
    )

    print(f"### Поток теста (iid) по языкам, `{keyset}`, thr\n")
    rows = [[f"`{c}`"] + [fmt(t(l, keyset, c)) for l in langs] for c in names]
    print(table(["кандидат"] + langs, rows))
    print(f"### Поток теста (iid) по языкам, `{keyset}`, lat\n")
    rows = [
        [f"`{c}`"] + [fmt(t(l, keyset, c, mode="lat")) for l in langs] for c in names
    ]
    print(table(["кандидат"] + langs, rows))

    print(f"## 4. Порог деления по длине ключа, `{keyset}`\n")
    for long in ("pilot", "ptrhash"):
        ladder = (
            [f"{long}+inline"]
            + [f"split{k}:{long}+ids" for k in SPLITS]
            + ["dacore+ids"]
        )
        heads = ["только хеш"] + [str(k) for k in SPLITS] + ["только DA"]
        for mode in ("thr", "lat"):
            rows, best_by_len = [], []
            for lang in langs:
                values = [t(lang, keyset, c, mode=mode) for c in ladder]
                if not any(values):
                    continue
                best = min((v, h) for v, h in zip(values, heads) if v)[1]
                s = prepared[lang][keyset]["stats"]
                singles = [
                    (t(lang, keyset, c, mode=mode), c) for c in names if is_single(c)
                ]
                singles = [
                    x
                    for x in singles
                    if x[0] and info(lang, keyset, x[1]).get("exact", True)
                ]
                single = min(singles) if singles else (None, "")
                split_best = (
                    min(v for v in values[1:-1] if v) if any(values[1:-1]) else None
                )
                rows.append(
                    [lang, fmt(s["key_len"]["mean_bytes"]), fmt(s["bytes_per_char"], 2)]
                    + [fmt(v) for v in values]
                    + [
                        best,
                        f"`{single[1]}` {fmt(single[0])}",
                        fmt(100 * (split_best / single[0] - 1))
                        if split_best and single[0]
                        else "—",
                    ]
                )
                best_by_len.append((s["key_len"]["mean_bytes"], best))
            print(f"Длинная сторона `{long}`, поток теста (iid), {mode}:\n")
            print(
                table(
                    ["язык", "ср. длина ключа", "байт/символ"]
                    + heads
                    + [
                        "лучший порог",
                        "лучшая одиночная (точная)",
                        "лучшее деление к ней, %",
                    ],
                    rows,
                )
            )

    print(f"## 5. Поток теста против равномерного по ключам, `{keyset}`\n")
    rows = []
    for lang in langs:
        both = [(c, t(lang, keyset, c), t(lang, keyset, c, "uni_hit")) for c in names]
        both = [b for b in both if b[1] and b[2]]
        if len(both) < 3:
            continue
        real_rank = {
            c: i for i, (c, _, _) in enumerate(sorted(both, key=lambda b: b[1]))
        }
        uni_rank = {
            c: i for i, (c, _, _) in enumerate(sorted(both, key=lambda b: b[2]))
        }
        n = len(both)
        rho = 1 - 6 * sum((real_rank[c] - uni_rank[c]) ** 2 for c in real_rank) / (
            n * (n * n - 1)
        )
        top_real = sorted(real_rank, key=real_rank.get)[:3]
        top_uni = sorted(uni_rank, key=uni_rank.get)[:3]
        mover = max(real_rank, key=lambda c: abs(real_rank[c] - uni_rank[c]))
        rows.append(
            [
                lang,
                fmt(rho, 2),
                ", ".join(f"`{c}`" for c in top_real),
                ", ".join(f"`{c}`" for c in top_uni),
                f"`{mover}`: {uni_rank[mover] + 1} → {real_rank[mover] + 1}",
            ]
        )
    print(
        table(
            [
                "язык",
                "ρ Спирмена",
                "первые три на потоке теста",
                "первые три на равномерном",
                "самый большой сдвиг (место uniform → поток)",
            ],
            rows,
        )
    )

    print(f"## 6. Фронт Парето «время — память», `{keyset}`, поток теста thr\n")
    front_count = defaultdict(int)
    rows = []
    for lang in langs:
        points = [
            (t(lang, keyset, c), info(lang, keyset, c).get("bytes_per_key"), c)
            for c in names
        ]
        points = [
            p
            for p in points
            if p[0]
            and p[1]
            and info(lang, keyset, p[2]).get("exact", True)
            and not p[2].startswith(("hashbrown", "call>"))
        ]
        front = [
            p
            for p in points
            if not any(
                q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1])
                for q in points
            )
        ]
        for p in front:
            front_count[p[2]] += 1
        rows.append(
            [
                lang,
                "; ".join(
                    f"`{c}` {fmt(ns_)} нс / {fmt(b)} Б" for ns_, b, c in sorted(front)
                ),
            ]
        )
    print(
        "Только точные кандидаты; `hashbrown` исключён как неотображаемый из файла.\n"
    )
    print(table(["язык", "недоминируемые (нс; Б/ключ)"], rows))
    print(
        table(
            ["кандидат", f"языков на фронте (из {len(langs)})"],
            [
                [f"`{c}`", n]
                for c, n in sorted(front_count.items(), key=lambda x: -x[1])
            ],
        )
    )

    print(f"## 7. Слой значений, `{keyset}`\n")
    rows = []
    for family, variants in (
        ("dacore", ["dacore+ids", "dacore+off", "dacore+offid", "dacore+rule"]),
        ("fst", ["fst+ids", "fst+rule"]),
        ("pilot", ["pilot+ids", "pilot+off", "pilot+inline", "pilot+rule"]),
        ("ptrhash", ["ptrhash+ids", "ptrhash+inline", "ptrhash+rule"]),
        (
            "split8:pilot",
            [
                "split8:pilot+ids",
                "split8:pilot+off",
                "split8:pilot+offid",
                "split8:pilot+rule",
            ],
        ),
        (
            "split8:ptrhash",
            ["split8:ptrhash+ids", "split8:ptrhash+off", "split8:ptrhash+rule"],
        ),
    ):
        base = variants[0]
        for cand in variants:
            ratios = {
                m: gm(
                    [
                        t(l, keyset, cand, mode=m) / t(l, keyset, base, mode=m)
                        for l in langs
                        if t(l, keyset, cand, mode=m) and t(l, keyset, base, mode=m)
                    ]
                )
                for m in ("thr", "lat")
            }
            rows.append(
                [
                    f"`{cand}`",
                    fmt(across(keyset, cand)),
                    fmt(across(keyset, cand, mode="lat")),
                    fmt(100 * (ratios["thr"] - 1)) if ratios["thr"] else "—",
                    fmt(100 * (ratios["lat"] - 1)) if ratios["lat"] else "—",
                    fmt(bytes_per_key(keyset, cand)),
                ]
            )
    print(
        table(
            [
                "кандидат",
                "поток thr",
                "поток lat",
                "к `+ids` thr, %",
                "к `+ids` lat, %",
                "Б/ключ",
            ],
            rows,
        )
    )

    print(f"## 8. Отрицательный фильтр (binary fuse 8), `{keyset}`\n")
    rows = []
    workloads = [MAIN, "iid_hit", "iid_miss"] + (
        ["cascade"] if keyset == "external" else []
    )
    for inner in ("dacore+ids", "fst+ids", "marisa+ids", "core+ids", "ptrhash+inline"):
        for lang in langs:
            cells = []
            for w in workloads:
                a, b = t(lang, keyset, f"bf8>{inner}", w), t(lang, keyset, inner, w)
                cells.append(fmt(100 * (a / b - 1)) if a and b else "—")
            miss = (
                1 - prepared[lang][keyset]["stats"]["stream"]["hit_rate"]
                if keyset in prepared.get(lang, {})
                else None
            )
            extra = (
                info(lang, keyset, f"bf8>{inner}").get("bytes_per_key", 0) or 0
            ) - (info(lang, keyset, inner).get("bytes_per_key", 0) or 0)
            rows.append(
                [f"`{inner}`", lang, fmt(100 * miss) if miss is not None else "—"]
                + cells
                + [fmt(extra, 2)]
            )
    print(
        "Изменение времени с фильтром к тому же кандидату без фильтра, % (минус — быстрее).\n"
    )
    print(
        table(
            ["структура", "язык", "промахов в потоке, %"] + workloads + ["+Б/ключ"],
            rows,
        )
    )

    print(f"## 9. Боевой кодек против лабораторных двойников, `{keyset}`\n")
    rows = []
    trio = [
        "core+ids",
        "call>split8:pilot+ids",
        "split8:pilot+ids",
        "split8:ptrhash+ids",
        "ptrhash+inline",
        "call>ptrhash+inline",
        "dacore+ids",
        "call>dacore+ids",
    ]
    for lang in langs:
        for cand in trio:
            c = counters.get((lang, keyset, cand, MAIN), {})
            rows.append(
                [
                    lang,
                    f"`{cand}`",
                    fmt(t(lang, keyset, cand)),
                    fmt(t(lang, keyset, cand, mode="lat")),
                    fmt(t(lang, keyset, cand, "iid_hit")),
                    fmt(t(lang, keyset, cand, "iid_miss")),
                    fmt(c.get("instructions")),
                    fmt(c.get("cycles")),
                    fmt(c.get("branch_misses"), 3),
                    fmt(c.get("l1d_miss"), 3),
                    fmt(c.get("cache_miss"), 3),
                    fmt(c.get("dtlb_miss"), 4),
                ]
            )
    print(
        "`call>` — тот же кандидат за настоящим вызовом функции (без инлайна в цикл): так боевой"
    )
    print("`HybridTable::get` и достаётся вызывающему.\n")
    print(
        table(
            [
                "язык",
                "кандидат",
                "поток thr",
                "поток lat",
                "попадания",
                "промахи",
                "инстр./запрос",
                "циклов/запрос",
                "промахов ветвлений",
                "промахов L1d",
                "cache-misses",
                "промахов dTLB",
            ],
            rows,
        )
    )

    print(f"## 10. Развёртка доли промахов, `{keyset}`, thr\n")
    points = ["000", "010", "025", "050", "075", "090", "100"]
    stream = {"000": "iid_hit", "100": "iid_miss"}
    rows = [
        [f"`{c}`"]
        + [fmt(across(keyset, c, stream.get(p, f"sweep_p{p}"))) for p in points]
        for c in names
    ]
    rows.sort(key=lambda r: float(r[1]) if r[1] != "—" else 1e9)
    print(table(["кандидат"] + [f"{int(p)} %" for p in points], rows))

    print(
        f"## 11. Счётчики на потоке теста, `{keyset}` (геом. среднее по языкам, на запрос)\n"
    )
    rows = []
    for cand in names:
        got = [counters.get((l, keyset, cand, MAIN), {}) for l in langs]
        cell = lambda name, d=2: fmt(gm([g.get(name) for g in got]), d)
        rows.append(
            [
                f"`{cand}`",
                cell("instructions", 1),
                cell("cycles", 1),
                cell("branch_misses", 3),
                cell("l1d_miss", 3),
                cell("cache_miss", 3),
                cell("llc_miss", 4),
                cell("dtlb_miss", 4),
            ]
        )
    print(
        table(
            [
                "кандидат",
                "инструкций",
                "циклов",
                "промахов ветвлений",
                "L1d",
                "cache-misses",
                "LLC",
                "dTLB",
            ],
            rows,
        )
    )

    print(f"## 12. Открытие, `{keyset}`\n")
    rows = []
    for cand in names:
        warm = [
            statistics.median(opens[(l, keyset, cand)]) / 1e3
            for l in langs
            if opens.get((l, keyset, cand))
        ]
        cold = [colds[(l, keyset, cand)] for l in langs if (l, keyset, cand) in colds]
        rows.append(
            [
                f"`{cand}`",
                fmt(statistics.median(warm)) if warm else "—",
                fmt(
                    statistics.median(
                        [c["open_and_first_answers_ns"] / 1e6 for c in cold]
                    ),
                    2,
                )
                if cold
                else "—",
                fmt(statistics.median([c["major_faults"] for c in cold]), 0)
                if cold
                else "—",
            ]
        )
    print(
        "Тёплое — повторное открытие (мкс, медиана по языкам; для кандидатов с кучей включает пересборку"
    )
    print(
        "или десериализацию). Холодное — файлы вытеснены из страничного кэша (`POSIX_FADV_DONTNEED`),"
    )
    print(
        "время от открытия до 64-го ответа реального потока (мс) и число major faults.\n"
    )
    print(
        table(
            [
                "кандидат",
                "тёплое открытие, мкс",
                "холодное до 64 ответов, мс",
                "major faults",
            ],
            rows,
        )
    )

print("## 12а. Артефакт дословного повтора теста\n")
print(
    "Тот же тест, проигранный дословно по кругу (`real`), против независимых выборок из него (`iid`):"
)
print(
    "короткую последовательность предсказатель переходов заучивает, и структура выглядит быстрее, чем на свежем тексте.\n"
)
rows = []
for keyset in keysets:
    for cand in (
        "dacore+ids",
        "core+ids",
        "split8:pilot+ids",
        "ptrhash+inline",
        "fst+ids",
        "marisa+ids",
        "sorted+ids",
    ):
        for lang in langs:
            a, b = t(lang, keyset, cand, "real"), t(lang, keyset, cand)
            ca, cb = (
                counters.get((lang, keyset, cand, "real"), {}),
                counters.get((lang, keyset, cand, MAIN), {}),
            )
            if a and b:
                rows.append(
                    [
                        keyset,
                        f"`{cand}`",
                        lang,
                        prepared[lang][keyset]["workloads"]["real"]["queries"],
                        fmt(a),
                        fmt(b),
                        fmt(100 * (b / a - 1)),
                        fmt(ca.get("branch_misses"), 3),
                        fmt(cb.get("branch_misses"), 3),
                    ]
                )
print(
    table(
        [
            "таблица",
            "кандидат",
            "язык",
            "токенов в тесте",
            "повтор thr",
            "iid thr",
            "iid к повтору, %",
            "промахов ветвлений, повтор",
            "промахов ветвлений, iid",
        ],
        rows,
    )
)

if census:
    print(f"## 13. Перепись: размер и сборка корпусной таблицы, {len(census)} языков\n")
    per = defaultdict(lambda: defaultdict(list))
    for lang, sets in census.items():
        body = sets.get("exact")
        if not body:
            continue
        n = body["stats"]["keys"]
        for cand, entry in body["candidates"].items():
            if entry["verified"]:
                per[cand]["bpk"].append(entry["bytes_per_key"])
                per[cand]["build"].append(entry["build_ns"] / n)
                per[cand]["peak"].append(entry["build_peak_heap"] / n)
    rows = []
    for cand, got in per.items():
        q = (
            statistics.quantiles(got["bpk"], n=4)
            if len(got["bpk"]) > 3
            else [min(got["bpk"])] * 3
        )
        rows.append(
            [
                f"`{cand}`",
                len(got["bpk"]),
                fmt(statistics.median(got["bpk"])),
                f"{fmt(q[0])}…{fmt(q[2])}",
                fmt(statistics.median(got["build"]), 0),
                fmt(statistics.median(got["peak"]), 0),
            ]
        )
    rows.sort(key=lambda r: float(r[2]))
    print(
        table(
            [
                "кандидат",
                "языков",
                "Б/ключ, медиана",
                "квартили",
                "сборка, нс/ключ",
                "пик кучи сборки, Б/ключ",
            ],
            rows,
        )
    )

mem = OUT / "memmodes.jsonl"
if mem.exists():
    print(
        "## 14. Где лежат байты: mmap против кучи, THP выключен и включён (таблица `exact`)\n"
    )
    got = defaultdict(list)
    for line in mem.read_text().splitlines():
        rec = json.loads(line)
        mode = ("куча" if rec.get("heap_images") else "mmap") + (
            ", THP выкл" if rec["thp_disabled"] else ", THP системный"
        )
        real = rec["results"].get(MAIN) or rec["results"]["real"]
        got[(rec["lang"], rec["cand"], mode)].append(
            (clock(real["thr"]), clock(real["lat"]), real["thr"].get("per_query", {}))
        )
    modes = [
        "mmap, THP выкл",
        "куча, THP выкл",
        "mmap, THP системный",
        "куча, THP системный",
    ]
    rows = []
    for lang, cand in sorted({(k[0], k[1]) for k in got}):
        base = got.get((lang, cand, modes[0]))
        for mode in modes:
            runs_ = got.get((lang, cand, mode))
            if not runs_:
                continue
            thr = settle([r[0] for r in runs_], NOISY)
            lat = settle([r[1] for r in runs_], NOISY)
            tlb = statistics.median(r[2].get("dtlb_miss", 0) for r in runs_)
            cyc = statistics.median(r[2].get("cycles", 0) for r in runs_)
            ref = settle([r[0] for r in base], NOISY) if base else None
            rows.append(
                [
                    lang,
                    f"`{cand}`",
                    mode,
                    fmt(thr),
                    fmt(lat),
                    fmt(cyc),
                    fmt(tlb, 4),
                    fmt(100 * (thr / ref - 1)) if ref else "—",
                ]
            )
    print(
        table(
            [
                "язык",
                "кандидат",
                "память",
                "поток thr",
                "поток lat",
                "циклов/запрос",
                "dTLB/запрос",
                "thr к «mmap, THP выкл», %",
            ],
            rows,
        )
    )

lengths = OUT / "lengths.jsonl"
if lengths.exists():
    print("## 15. Профиль по длине ключа: где пересекаются обход DA и хеш\n")
    print(
        "Попадания реального потока, разложенные по длине ключа в байтах (корзины от 200 запросов)."
    )
    print(
        "Пересечение — наименьшая длина, начиная с которой хеш не медленнее DA на всех более длинных корзинах.\n"
    )
    prof = defaultdict(list)
    for line in lengths.read_text().splitlines():
        rec = json.loads(line)
        if str(rec.get("round")) in SKIP:
            continue
        for workload, modes in rec["results"].items():
            for mode, got in modes.items():
                prof[
                    (rec["lang"], rec["keyset"], rec["cand"], int(workload[4:]), mode)
                ].append(clock(got))
    prof = {k: settle(v, NOISY) for k, v in prof.items()}
    pairs = [
        ("dacore+ids", "pilot+inline"),
        ("dacore+ids", "ptrhash+inline"),
        ("dacore+ids", "pilot+ids"),
        ("dacore+off", "pilot+inline"),
    ]
    for keyset in sorted({k[1] for k in prof}):
        rows = []
        for lang in sorted({k[0] for k in prof if k[1] == keyset}):
            row = [lang]
            for mode in ("thr", "lat"):
                for da, hashed in pairs:
                    lens = sorted(
                        {
                            k[3]
                            for k in prof
                            if k[:3] == (lang, keyset, da) and k[4] == mode
                        }
                        & {
                            k[3]
                            for k in prof
                            if k[:3] == (lang, keyset, hashed) and k[4] == mode
                        }
                    )
                    cross = None
                    for length in reversed(lens):
                        if (
                            prof[(lang, keyset, hashed, length, mode)]
                            <= prof[(lang, keyset, da, length, mode)]
                        ):
                            cross = length
                        else:
                            break
                    row.append(
                        "—"
                        if not lens
                        else (f">{lens[-1]}" if cross is None else str(cross))
                    )
            rows.append(row)
        heads = [f"{mode}: `{a}`×`{b}`" for mode in ("thr", "lat") for a, b in pairs]
        print(f"Таблица `{keyset}`: длина пересечения, байт.\n")
        print(table(["язык"] + heads, rows))
        for mode in ("thr", "lat"):
            print(
                f"Нс на попадание по длине ключа, `{keyset}`, {mode} (геом. среднее по языкам):\n"
            )
            cands_ = sorted({k[2] for k in prof if k[1] == keyset})
            all_lens = sorted({k[3] for k in prof if k[1] == keyset})
            rows = [
                [f"`{c}`"]
                + [
                    fmt(
                        gm(
                            [
                                v
                                for k, v in prof.items()
                                if k[1:] == (keyset, c, n, mode)
                            ]
                        )
                    )
                    for n in all_lens
                ]
                for c in cands_
            ]
            print(table(["кандидат"] + [str(n) for n in all_lens], rows))
