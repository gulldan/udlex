#!/usr/bin/env bash
# GPU cells of the all-language C9 run, after its CPU phases: vLLM is stopped for this phase
# only (the owner allowed about three hours, 20.11.2026) and started again on any exit.
#   gpu_phase.sh OUT              the stand as in run.sh
set -uo pipefail
RUN=${UDLEX_SPEED_RUN:?set UDLEX_SPEED_RUN to the run directory}
H=$RUN/harness/speed
PY=$RUN/.venv-udlex/bin/python
LOCK=${UDLEX_SPEED_LOCK:?set UDLEX_SPEED_LOCK to the exclusive lock file of the stand}
QUIET=${UDLEX_SPEED_PAUSE:-}  # containers to stop (vLLM), space-separated
OUT=$1
LIMIT=${LIMIT:-12600}  # 3.5 hours, then vLLM comes back whatever is left
say() { echo "$(TZ=Asia/Tokyo date '+%F %T %Z') $*" | tee -a "$OUT/vllm.log"; }
say "queued; waiting for the CPU phases"
until grep -q ALL-LANGUAGES-DONE "$OUT.log" 2>/dev/null; do sleep 60; done
exec 9>"$LOCK"; flock 9
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
        say "$c started: ${state:-NOT RUNNING}"
    done
    stopped=0
}
trap restore EXIT
trap 'exit 130' INT TERM HUP
for c in $QUIET; do say "$c before: $(docker ps --filter "name=^/$c\$" --format '{{.Status}}')"; done
stopped=1
for c in $QUIET; do
    docker stop "$c" >/dev/null
    say "$c stopped: $(docker inspect -f '{{.State.Status}}' "$c")"
done
OPTS=$("$PY" -I -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["options"]))' "$RUN/receipts/auto-20261119T081436Z/sweep/chosen.json")
# shellcheck disable=SC2046,SC2086  # languages and options are word lists
timeout "$LIMIT" taskset -c 16-31 "$PY" -I "$H/bench.py" run "$RUN" "$OUT" \
    --langs $(cat "$OUT/languages.txt") --rounds 3 --windows 3 --cpu1 4 $OPTS --stanza-gpu-tokens 0 \
    --systems stanza trankit spacy-trf --modes gpu \
    --note "C9 on all 169 languages, GPU phase; vLLM stopped, times in vllm.log" \
    >>"$OUT/run.log" 2>&1 || say "GPU phase: bench.py exited with $?"
restore
"$PY" -I "$H/bench.py" summary "$OUT" > "$OUT/summary.log" 2>&1 || say "no summary"
say "GPU-PHASE-DONE $OUT/SUMMARY.md"
