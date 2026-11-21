#!/usr/bin/env bash
# The encoding lab's measurement, one command.
#
#   ./run_final.sh                 final: detaches into tmux
#                                  `encoding-final-<stamp>`, holds the stand's
#                                  exclusive lock, refuses a busy machine
#                                  (FORCE=1 overrides; FORCE_WHY goes into
#                                  the receipt)
#   MODE=screen ./run_final.sh     screening on a shared machine: no lock, one
#                                  round, short samples, receipts marked noisy
#
# Knobs: LANGS, KEYSETS, CANDS (comma list), ROUNDS, CPU, CENSUS=0|1, WORK,
# MEMMODES=0|1 with MEM_LANGS and MEM_CANDS, LEN_CANDS=none to skip, LOCK.
# An extra round into an existing receipt: STAMP=<its stamp> ROUND_FIRST=4
# ROUNDS=4 COLD=0 (counters again, no cold start); logs are appended.
# Machine load is sampled every minute into load.jsonl beside the receipts.
# Receipts land in $RUN/receipts/<mode>-<UTC stamp>/.
set -euo pipefail

RUN=${RUN:-/mnt/hdd-4tb/codex-runs/udlex-encoding-20261119}
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${MODE:-final}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$RUN/receipts/$MODE-$STAMP
LOCK=${LOCK:-/mnt/hdd-2tb/lemmatest-perf-results/.exclusive.lock}

if [[ ${1:-} != --inner ]]; then
    read -r load _ </proc/loadavg
    if [[ $MODE == final && ${FORCE:-0} != 1 ]] && awk -v l="$load" 'BEGIN { exit !(l > 2.0) }'; then
        echo "load average $load: the machine is busy; final times would be noise (FORCE=1 to override)" >&2
        exit 1
    fi
    mkdir -p "$OUT"
    lock=""
    [[ $MODE == final ]] && lock="flock $LOCK"
    pass="MODE=$MODE STAMP=$STAMP RUN=$RUN"
    for knob in LANGS KEYSETS CANDS ROUNDS ROUND_FIRST COLD CPU CENSUS FORCE FORCE_WHY SAMPLE_MS SAMPLES WORK MEMMODES MEM_LANGS MEM_CANDS LEN_CANDS DICTS REPO INPUTS; do
        [[ -n ${!knob:-} ]] && pass+=" $knob='${!knob}'"
    done
    tmux new-session -d -s "encoding-$MODE-$STAMP" \
        "env $pass $lock $HERE/run_final.sh --inner 2>&1 | tee -a $OUT/run.log"
    echo "started tmux session encoding-$MODE-$STAMP; receipts: $OUT"
    exit 0
fi

export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
INPUTS=${INPUTS:-/mnt/hdd-4tb/codex-runs/udlex-compare-20261119/census/inputs/order}
# Dictionary tables come from the shipped 1.0.0 bundles, unpacked here as
# <iso>/current + .generations/; REPO instead reads the inventory's sources.
DICTS=${DICTS:-$RUN/dicts-1.0.0/bundle}
if [[ -n ${REPO:-} ]]; then dicts=(--repo "$REPO"); else dicts=(--dicts "$DICTS"); fi
# Scripts and sizes: Latin, Cyrillic, Greek, Arabic, Devanagari, Hangul,
# kana/kanji, hanzi; agglutinative fi/tr/hu; low-resource wol.
LANGS=${LANGS:-eng rus deu fin tur arb hin kor jpn cmn ces hun ell wol}
KEYSETS=${KEYSETS:-exact external}
CENSUS=${CENSUS:-1}

# The least busy CPU right now; times stay noisy, counters less so.
idle_cpu() {
    python3 - <<'EOF'
import time
def snap():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            v = list(map(int, line.split()[1:]))
            out[int(line.split()[0][3:])] = (sum(v), v[3] + v[4])
    return out
a = snap(); time.sleep(3); b = snap()
print(max(a, key=lambda c: (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0])))
EOF
}

if [[ $MODE == final ]]; then
    ROUNDS=${ROUNDS:-3} SAMPLE_MS=${SAMPLE_MS:-30} SAMPLES=${SAMPLES:-5}
    CPU=${CPU:-6}
    WORK=${WORK:-$RUN/work-final-$STAMP}
    NOTE=quiet
else
    ROUNDS=${ROUNDS:-1} SAMPLE_MS=${SAMPLE_MS:-15} SAMPLES=${SAMPLES:-3}
    CPU=${CPU:-$(idle_cpu)}
    WORK=${WORK:-$RUN/work}
    NOTE=noisy
fi
mkdir -p "$OUT/prepare" "$OUT/census" "$WORK"
python3 "$HERE/load_sampler.py" "$OUT/load.jsonl" "$CPU" 60 &
sampler=$!
trap 'kill "$sampler" 2>/dev/null' EXIT

cd "$HERE"
cargo build --release
cargo test --release
BIN=$HERE/target/release/encoding-lab
CANDS=${CANDS:-$($BIN list | paste -sd, -)}

{
    echo "--- $(date -u +%Y-%m-%dT%H:%M:%SZ) rounds ${ROUND_FIRST:-1}..$ROUNDS cold=${COLD:-1}"
    echo "mode: $MODE  stamp: $STAMP  cpu: $CPU (SMT siblings $(cat /sys/devices/system/cpu/cpu"$CPU"/topology/thread_siblings_list))  rounds: $ROUNDS  sample: ${SAMPLES}x${SAMPLE_MS}ms"
    echo "keysets: $KEYSETS  census: $CENSUS  memmodes: ${MEMMODES:-1}"
    echo "force: ${FORCE:-0}${FORCE_WHY:+ ($FORCE_WHY)}"
    echo "host: $(uname -srm)"
    echo "cpu model: $(lscpu | sed -n 's/^Model name: *//p')"
    echo "caches: $(lscpu | grep -E '^L(1d|2|3) cache' | tr -s ' ' | paste -sd';' -)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu"$CPU"/cpufreq/scaling_governor)"
    echo "thp: $(cat /sys/kernel/mm/transparent_hugepage/enabled) (the lab disables it for itself)"
    echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
    echo "loadavg at start: $(cat /proc/loadavg)"
    echo "storage: $(df --output=source,fstype "$WORK" | tail -1)"
    echo "rustc: $(rustc --version)"
    echo "binary sha256: $(sha256sum "$BIN" | cut -d' ' -f1)"
    echo "core hybrid.rs sha256: $(sha256sum "$HERE/../../../../rust/src/lexicon/table/hybrid.rs" | cut -d' ' -f1)"
    echo "inputs: $INPUTS"
    echo "dictionaries: ${REPO:-$DICTS}"
    echo "crates:"
    for crate in crawdad fst hashbrown ph ptr_hash rsmarisa sux xorf yada memmap2 rayon; do
        awk -v c="$crate" '$0 == "name = \"" c "\"" { getline; gsub(/version = |"/, ""); print "  " c " " $0 }' Cargo.lock | sort -u
    done
} | tee -a "$OUT/environment.txt"

for lang in $LANGS; do
    if [[ ! -f $WORK/$lang/prepare.json ]]; then
        taskset -c "$CPU" "$BIN" prepare --inputs "$INPUTS" "${dicts[@]}" --work "$WORK" \
            --lang "$lang" --keysets "${KEYSETS// /,}" --cands "$CANDS" \
            || { echo "prepare $lang failed" | tee -a "$OUT/failures.log"; continue; }
    fi
    cp "$WORK/$lang/prepare.json" "$OUT/prepare/$lang.json"
done

read -ra names <<<"${CANDS//,/ }"
for round in $(seq "${ROUND_FIRST:-1}" "$ROUNDS"); do
    extra=""
    # Counters (and a cold start) in the first round measured.
    if [[ $round == "${ROUND_FIRST:-1}" ]]; then
        extra="--counters"
        [[ ${COLD:-1} == 1 ]] && extra+=" --cold"
    fi
    for lang in $LANGS; do
        for keyset in $KEYSETS; do
            [[ -d $WORK/$lang/$keyset ]] || continue
            # Rotate the order so no candidate always follows the same one.
            for i in "${!names[@]}"; do
                cand=${names[$(((i + round * 7) % ${#names[@]}))]}
                # shellcheck disable=SC2086
                taskset -c "$CPU" "$BIN" bench --work "$WORK" --lang "$lang" --keyset "$keyset" \
                    --cand "$cand" --sample-ms "$SAMPLE_MS" --samples "$SAMPLES" \
                    --note "$NOTE" --round "$round" $extra >>"$OUT/bench.jsonl" \
                    || echo "bench $lang/$keyset/$cand round $round failed" | tee -a "$OUT/failures.log"
            done
            echo "round $round $lang/$keyset done ($(date -u +%H:%M:%S), load $(cut -d' ' -f1 /proc/loadavg))"
        done
    done
done

# The per-length profile: stream hits bucketed by byte length, for the two
# sides of the hybrid, to read the crossover off directly.
for round in $(seq "${ROUND_FIRST:-1}" "$ROUNDS"); do
    for lang in $LANGS; do
        for keyset in $KEYSETS; do
            [[ -d $WORK/$lang/$keyset ]] || continue
            for cand in ${LEN_CANDS:-dacore+ids dacore+off pilot+inline pilot+ids ptrhash+inline ptrhash+ids fst+ids}; do
                [[ $cand == none ]] && continue
                taskset -c "$CPU" "$BIN" bench --work "$WORK" --lang "$lang" --keyset "$keyset" \
                    --cand "$cand" --sample-ms "$SAMPLE_MS" --samples "$SAMPLES" --lengths \
                    --note "$NOTE" --round "$round" >>"$OUT/lengths.jsonl" \
                    || echo "lengths $lang/$keyset/$cand failed" | tee -a "$OUT/failures.log"
            done
        done
    done
done

# Where the bytes live: the same images mapped or copied to the heap, with
# THP off or as the system sets it. Separates the codec from its memory.
if [[ ${MEMMODES:-1} == 1 ]]; then
    for round in $(seq "${ROUND_FIRST:-1}" "$ROUNDS"); do
        for lang in ${MEM_LANGS:-rus kor tur deu eng}; do
            [[ -d $WORK/$lang/exact ]] || continue
            for cand in ${MEM_CANDS:-core+ids call>split8:pilot+ids split8:pilot+ids split8:ptrhash+ids ptrhash+inline dacore+ids}; do
                for mode in "" "ENCODING_LAB_HEAP=1" "ENCODING_LAB_THP=1" "ENCODING_LAB_HEAP=1 ENCODING_LAB_THP=1"; do
                    # shellcheck disable=SC2086
                    env $mode taskset -c "$CPU" "$BIN" bench --work "$WORK" --lang "$lang" --keyset exact \
                        --cand "$cand" --sample-ms "$SAMPLE_MS" --samples "$SAMPLES" --counters \
                        --note "$NOTE" --round "$round" >>"$OUT/memmodes.jsonl" \
                        || echo "memmodes $lang/$cand [$mode] failed" | tee -a "$OUT/failures.log"
                done
            done
        done
    done
fi

# Size and build cost of the corpus table on every language that has both splits.
if [[ $CENSUS == 1 ]]; then
    for dir in "$INPUTS"/*/; do
        lang=$(basename "$dir")
        [[ -s $dir/train.jsonl && -s $dir/test.jsonl ]] || continue
        # shellcheck disable=SC2015  # a failed copy is a failed language too
        taskset -c "$CPU" "$BIN" prepare --inputs "$INPUTS" "${dicts[@]}" --work "$WORK-census" \
            --lang "$lang" --keysets exact --cands "$CANDS" --discard \
            --uniform-queries 20000 --mixed-queries 5000 2>>"$OUT/census.log" \
            && cp "$WORK-census/$lang/prepare.json" "$OUT/census/$lang.json" \
            || echo "census $lang failed" | tee -a "$OUT/failures.log"
    done
fi

echo "loadavg at end: $(cat /proc/loadavg)" | tee -a "$OUT/environment.txt"
python3 "$HERE/summarize.py" "$OUT" >"$OUT/SUMMARY.md"
echo "done: $OUT/SUMMARY.md"
