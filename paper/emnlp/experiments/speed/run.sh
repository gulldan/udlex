#!/usr/bin/env bash
# Speed and memory of udlex 1.0.0 against the compared lemmatizers (task C9).
#
#   speed/run.sh ship     (laptop) copy the harness and udlex at $COMMIT to $UDLEX_SPEED_HOST
#   run.sh setup          (server) environments, dictionary bundles, udlex reference answers
#   run.sh smoke          (server) every code path on two CPUs, no lock: times are not results
#   run.sh auto           (server) the whole measurement unattended, in tmux `speed-auto-<stamp>`:
#                         batch sweep -> choice -> dry run (eng, kor) -> final (12 languages)
#                         -> SUMMARY.md; receipts in $RUN/receipts/auto-<stamp>/
#   run.sh final          (server) the final alone (vLLM stopped meanwhile), ARGS = batch options
#   run.sh retry BASE     (server) after auto: time the failed cells again (vLLM off meanwhile)
#   run.sh summary OUT    (server) tables of a finished run
#
# Every timed step waits for the stand's exclusive lock; no timing runs while the owner's
# dictionary build is up ($REPRO exists or tmux dicts-repro lives).
#
# The stand, from the environment: UDLEX_SPEED_RUN the run directory, UDLEX_SPEED_LOCK the
# exclusive lock file, UDLEX_SPEED_COMPARE the quality comparison, UDLEX_SPEED_DICTS the
# dictionary build; optional UDLEX_SPEED_PAUSE, UDLEX_SPEED_INPUTS, REPRO, ORACLE (auto).
set -euo pipefail

RUN=${UDLEX_SPEED_RUN:?set UDLEX_SPEED_RUN to the run directory}
H=$(cd "$(dirname "$0")" && pwd)
LOCK=${UDLEX_SPEED_LOCK:?set UDLEX_SPEED_LOCK to the exclusive lock file of the stand}
C=${UDLEX_SPEED_COMPARE:?set UDLEX_SPEED_COMPARE to the quality comparison}
D=${UDLEX_SPEED_DICTS:?set UDLEX_SPEED_DICTS to the dictionary build}
PY=$RUN/.venv-udlex/bin/python
LANGS=${LANGS:-eng rus deu fin tur arb hin kor jpn ces hun gla}
REPRO=${REPRO:-}  # a flag file that exists while the owner's dictionary build runs
PAUSE=(--pause-tmux dicts-repro)
[[ -z $REPRO ]] || PAUSE=(--pause-while "$REPRO" "${PAUSE[@]}")
owner_busy() { [[ -n $REPRO && -e $REPRO ]] || tmux has-session -t dicts-repro 2>/dev/null; }
say() { echo "$(TZ=Asia/Tokyo date '+%F %T %Z') $*"; }  # the owner reads JST
ORACLE=${ORACLE:-}  # the engine's lemma oracle report, required by auto
# A tmux session starts from the tmux server's environment: hand it the stand.
stand() {
    local v
    printf env
    for v in UDLEX_SPEED_RUN UDLEX_SPEED_LOCK UDLEX_SPEED_COMPARE UDLEX_SPEED_DICTS \
        UDLEX_SPEED_INPUTS UDLEX_SPEED_PAUSE REPRO ORACLE; do
        [[ -z ${!v:-} ]] || printf ' %q' "$v=${!v}"
    done
}
QUIET=${UDLEX_SPEED_PAUSE:-}  # containers stopped for the final only (vLLM), space-separated

case ${1:-} in
ship)
    SRC=${UDLEX_SRC:?set UDLEX_SRC to a udlex checkout}
    HOST=${UDLEX_SPEED_HOST:?set UDLEX_SPEED_HOST to the stand, as ssh knows it}
    COMMIT=${COMMIT:-4107fcd}
    E=$(cd "$H/.." && pwd)
    # shellcheck disable=SC2029  # RUN names the server's directory: expanded here
    ssh "$HOST" "mkdir -p $RUN/harness $RUN/src-udlex"
    # shellcheck disable=SC2029  # RUN names the server's directory: expanded here
    git -C "$SRC" archive "$COMMIT" | ssh "$HOST" "tar -x -C $RUN/src-udlex"
    # shellcheck disable=SC2029  # RUN names the server's directory: expanded here
    git -C "$SRC" rev-parse "$COMMIT" | ssh "$HOST" "cat > $RUN/src-udlex/COMMIT"
    rsync -a --exclude target --exclude __pycache__ --exclude Cargo.lock \
        "$H" "$E/compare_engines.py" "$E/order_summary.py" "$E/verify_snapshot.py" \
        "$HOST:$RUN/harness/"
    ;;
setup)
    export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
    export CARGO_TARGET_DIR=$RUN/target
    cd "$RUN"
    # The wheel and the native worker: one toolchain, one release profile (LTO, 1 unit).
    RUSTUP_TOOLCHAIN=$(sed -n 's/^channel = "\(.*\)"/\1/p' src-udlex/rust-toolchain.toml)
    export RUSTUP_TOOLCHAIN
    [[ -x $PY ]] || uv venv -q --python 3.12 .venv-udlex
    VIRTUAL_ENV=$RUN/.venv-udlex uv pip install -q maturin==1.16.0
    (cd src-udlex && nice "$RUN/.venv-udlex/bin/maturin" build -q --release -i "$PY" -o "$RUN/wheels")
    VIRTUAL_ENV=$RUN/.venv-udlex uv pip install -q --reinstall --no-deps "$RUN"/wheels/udlex-1.0.0-*.whl
    [[ -f $H/native/Cargo.lock ]] || cp src-udlex/rust/Cargo.lock "$H/native/Cargo.lock"
    nice cargo build -q --release --manifest-path "$H/native/Cargo.toml"
    # UDPipe 1 locally, with the UD 2.5 models the census scored through LINDAT.
    [[ -x .venv-udpipe1/bin/python ]] || uv venv -q --python 3.12 .venv-udpipe1
    VIRTUAL_ENV=$RUN/.venv-udpipe1 uv pip install -q ufal.udpipe==1.4.0.1
    mkdir -p udpipe1-models
    # shellcheck disable=SC2086  # LANGS is a word list, split on purpose
    for m in $("$PY" -I -c 'import json,sys; c=json.load(open(sys.argv[1]))["cells"]; print(" ".join(c[i]["model"] for i in sys.argv[2:] if c.get(i,{}).get("status")=="scored"))' "$C/results-census/udpipe1.json" $LANGS); do
        [[ -s udpipe1-models/$m.udpipe ]] ||
            curl -fsSL -o "udpipe1-models/$m.udpipe" "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/$m.udpipe"
    done
    # CuPy for spaCy on the GPU, beside (not inside) the shared spaCy environment.
    VIRTUAL_ENV=$RUN/.venv-udlex uv pip install -q --target pylib-cupy --no-deps cupy-cuda13x==14.2.0 fastrlock ||
        echo "no cupy: spaCy GPU cells will fail"
    # Bundles and reference answers, with the Python that scored the dictionaries.
    # shellcheck disable=SC2086  # LANGS is a word list, split on purpose
    "$D/.venv/bin/python" -I "$H/bench.py" prepare "$RUN" $LANGS
    echo SETUP-DONE
    ;;
smoke)
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    OUT=$RUN/receipts/smoke-$STAMP
    mkdir -p "$OUT"
    # Beside someone else's locked run: low priority, tiny windows, two CPUs of CCD1.
    tmux new-session -d -s "speed-smoke-$STAMP" \
        "$(stand) nice -n 19 taskset -c 24-27 $PY -I $H/bench.py run $RUN $OUT --langs eng kor --rounds 1 --windows 1 --target 0.5 --warm 0.5 --cpu1 9 --cpus16 9 10 ${ARGS:-} 2>&1 | tee $OUT/run.log; $PY -I $H/bench.py summary $OUT > $OUT/summary.log 2>&1"
    echo "tmux speed-smoke-$STAMP; receipts: $OUT"
    ;;
final)
    # The final alone (vLLM off, as in auto); ARGS: batch options, e.g. from bench.py choose.
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    BASE=$RUN/receipts/final-$STAMP
    mkdir -p "$BASE"
    tmux new-session -d -s "speed-final-$STAMP" \
        "$(stand) flock $LOCK bash $0 final-inner $BASE ${ARGS:-} 2>&1 | tee -a $BASE/auto.log; $PY -I $H/bench.py summary $BASE/final > $BASE/summary.log 2>&1"
    echo "tmux speed-final-$STAMP; receipts: $BASE/final"
    ;;
auto)
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    BASE=$RUN/receipts/auto-$STAMP
    mkdir -p "$BASE"
    tmux new-session -d -s "speed-auto-$STAMP" "$(stand) bash $0 auto-inner $BASE 2>&1 | tee -a $BASE/auto.log"
    echo "tmux speed-auto-$STAMP; receipts: $BASE"
    ;;
auto-inner)
    BASE=$2
    say "udlex: $(cat "$RUN/src-udlex/COMMIT"); checking the installed build against ref/udlex"
    if ! "$PY" -I "$H/bench.py" verify "$RUN" >"$BASE/verify.txt" 2>&1; then
        say "STOPPED: udlex answers differ from the references ($BASE/verify.txt)"
        exit 1
    fi
    say "sweep: waiting for the lock"
    flock "$LOCK" bash "$0" sweep-inner "$BASE/sweep" || say "sweep failed; the final keeps the defaults"
    OPTS=$("$PY" -I "$H/bench.py" choose "$BASE/sweep")
    say "chosen: $OPTS"
    say "dry: waiting for the dictionary build to end"
    while owner_busy; do sleep 60; done
    say "dry: waiting for the lock"
    # shellcheck disable=SC2086
    flock "$LOCK" taskset -c 16-31 "$PY" -I "$H/bench.py" run "$RUN" "$BASE/dry" \
        --langs eng kor --rounds 1 --windows 2 $OPTS "${PAUSE[@]}" \
        >"$BASE/dry.log" 2>&1 || say "dry: bench.py exited with $?"
    "$PY" -I "$H/bench.py" summary "$BASE/dry" >/dev/null || say "dry: no summary"
    if ! "$PY" -I "$H/bench.py" check "$BASE/dry" >"$BASE/dry-check.txt"; then
        say "STOPPED before the final: the dry run is unfit ($BASE/dry-check.txt)"
        exit 1
    fi
    differ=$(sed -n 's/.*ORACLE same=[0-9]* differ=\([0-9]*\).*/\1/p' "$ORACLE" 2>/dev/null | tail -1)
    if [[ $differ != 0 ]]; then
        say "STOPPED before the final: the lemma oracle of the engine says differ=${differ:-missing} ($ORACLE)"
        exit 1
    fi
    say "oracle: $(grep ORACLE "$ORACLE")"
    say "final: waiting for the dictionary build to end"
    while owner_busy; do sleep 60; done
    say "final: waiting for the lock"
    # shellcheck disable=SC2086
    flock "$LOCK" bash "$0" final-inner "$BASE" $OPTS || say "final: exited with $?"
    "$PY" -I "$H/bench.py" summary "$BASE/final" >/dev/null || say "final: no summary"
    ln -sfn "$BASE/final" "$RUN/receipts/LATEST-FINAL"
    say "AUTO-DONE: $BASE/final/SUMMARY.md"
    ;;
final-inner)
    # Holding the lock: vLLM off for the whole final (GPU cells are spread over every
    # round), back on by the EXIT trap, also after an error or a signal.
    BASE=$2
    shift 2
    mkdir -p "$BASE/final"
    note() { say "$*" | tee -a "$BASE/final/vllm.log"; }
    stopped=0
    restore() {
        [[ $stopped == 1 ]] || return 0
        for c in $QUIET; do
            for _ in 1 2 3; do
                docker start "$c" >/dev/null 2>&1 || true
                sleep 15
                state=$(docker ps --filter "name=^/$c\$" --format '{{.Status}}')
                [[ $state == Up* ]] && break
            done
            note "$c started: ${state:-NOT RUNNING}"
        done
        stopped=0
    }
    trap restore EXIT
    trap 'exit 130' INT TERM HUP
    for c in $QUIET; do note "$c before: $(docker ps --filter "name=^/$c\$" --format '{{.Status}}')"; done
    stopped=1
    for c in $QUIET; do
        docker stop "$c" >/dev/null
        note "$c stopped: $(docker inspect -f '{{.State.Status}}' "$c")"
    done
    extra=()
    [[ ${RETRY:-0} == 1 ]] && extra=(--retry-failed)  # same cells, rounds and order
    # shellcheck disable=SC2086  # LANGS is a word list, split on purpose
    taskset -c 16-31 "$PY" -I "$H/bench.py" run "$RUN" "$BASE/final" --langs $LANGS \
        --rounds 5 --windows 3 "$@" --stanza-gpu-tokens 0 "${PAUSE[@]}" "${extra[@]}" \
        --note "containers ${QUIET:-none} stopped for the final, times in vllm.log; udlex $(cat "$RUN/src-udlex/COMMIT")" \
        >>"$BASE/final.log" 2>&1 || note "bench.py exited with $?"
    ;;
retry)
    # Time again the failed cells of a finished auto run, vLLM off meanwhile.
    BASE=$2
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    tmux new-session -d -s "speed-retry-$STAMP" "$(stand) bash $0 retry-inner $BASE 2>&1 | tee -a $BASE/auto.log"
    echo "tmux speed-retry-$STAMP; receipts: $BASE/final"
    ;;
retry-inner)
    BASE=$2
    say "retry: waiting for the auto run to finish"
    until grep -qE "AUTO-DONE|STOPPED" "$BASE/auto.log"; do sleep 60; done
    count() { "$PY" -I -c 'import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; ok={(x["lang"],x["system"],x["mode"],x["round"]) for x in r if x["status"]=="ok"}; print(sum(x["status"]=="failed" and (x["lang"],x["system"],x["mode"],x["round"]) not in ok for x in r))' "$BASE/final/cells.jsonl"; }
    left=$(count)
    if [[ $left == 0 ]]; then
        say "retry: no failed cells"
    else
        OPTS=$("$PY" -I -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["options"]))' "$BASE/sweep/chosen.json")
        langs=$("$PY" -I -c 'import json,sys; print(" ".join(sorted({json.loads(l)["lang"] for l in open(sys.argv[1]) if json.loads(l)["status"]=="failed"})))' "$BASE/final/cells.jsonl")
        say "retry: $left failed cells ($langs); waiting for the lock"
        # shellcheck disable=SC2086
        LANGS=$langs RETRY=1 flock "$LOCK" bash "$0" final-inner "$BASE" $OPTS || say "retry: exited with $?"
        say "retry: $(count) cells still failed"
    fi
    "$PY" -I "$H/bench.py" summary "$BASE/final" >/dev/null || say "retry: no summary"
    say "RETRY-DONE: $BASE/final/SUMMARY.md"
    ;;
sweep-inner)
    # Batch sizes on eng, one round; bench.py choose picks the fastest of each cell.
    OUT=$2
    if owner_busy; then
        # The build holds CPU 0-7,16-23 and the encoding lab 12/28: the rest of CCD1 only.
        DRV=24-27,29-31
        CPUS=(--cpu1 9 --cpus16 8 9 10 11 13 14 15)
    else
        DRV=16-31 CPUS=()
    fi
    note() {
        if owner_busy; then
            echo "dictionary build dicts-repro running (CPU 0-7,16-23, first L3) at the start of this step; times for the choice of batch sizes only"
        else
            echo "no dictionary build at the start of this step"
        fi
    }
    B=(taskset -c "$DRV" "$PY" -I "$H/bench.py" run "$RUN")
    S=(--langs eng --rounds 1 --windows 3 "${CPUS[@]}")
    for b in 1 10 100 1000 10000; do
        "${B[@]}" "$OUT/udlex-batch-$b" "${S[@]}" --note "$(note)" --systems udlex-py --modes 1c thr16 --batch "$b"
    done
    for n in 16 64 256 1024; do
        "${B[@]}" "$OUT/neural-$n" "${S[@]}" --note "$(note)" --systems stanza trankit spacy-trf --modes thr16 gpu \
            --neural-batch "$n" --stanza-gpu-tokens 2000
    done
    for t in 5000 1000; do
        "${B[@]}" "$OUT/stanza-gpu-$t" "${S[@]}" --note "$(note)" --systems stanza --modes gpu --neural-batch 1024 \
            --stanza-gpu-tokens "$t"
    done
    for d in "$OUT"/*/; do "$PY" -I "$H/bench.py" summary "$d" >/dev/null || true; done
    echo SWEEP-DONE
    ;;
summary)
    "$PY" -I "$H/bench.py" summary "$2"
    ;;
*)
    sed -n '2,19p' "$0"
    exit 2
    ;;
esac
