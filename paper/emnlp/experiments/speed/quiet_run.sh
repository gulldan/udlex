#!/usr/bin/env bash
# C9 again on a quiet machine (the owner allowed it, 21.11.2026): after the current GPU phase,
# stop vLLM and the other services of UDLEX_SPEED_PAUSE, time every system on all 169 languages
# (udlex on the hybrid release), then start exactly the stopped containers again.
# The stand as in run.sh; UDLEX_SPEED_RELEASE is the hybrid release, UDLEX_SPEED_UD the
# UD 2.19 treebanks for fill_inputs.py.
set -u
S=${UDLEX_SPEED_RUN:?set UDLEX_SPEED_RUN to the run directory}
H=$S/harness/speed
REL=${UDLEX_SPEED_RELEASE:?set UDLEX_SPEED_RELEASE to the release to time}
LOCK=${UDLEX_SPEED_LOCK:?set UDLEX_SPEED_LOCK to the exclusive lock file of the stand}
OUT=${OUT:-$S/receipts/quiet-$(date -u +%Y%m%dT%H%M%SZ)}  # an existing OUT resumes its cells.jsonl
mkdir -p "$OUT"
export UDLEX_SPEED_INPUTS=$S/inputs-order  # census inputs + UD test text for bul, fro, tha
"$S/.venv-udlex/bin/python" -I "$H/fill_inputs.py" "$UDLEX_SPEED_INPUTS" > "$OUT/inputs.log" 2>&1 || exit 1
say() { echo "$(TZ=Asia/Tokyo date "+%F %T %Z") $*" | tee -a "$OUT/services.log"; }
QUIET=${UDLEX_SPEED_PAUSE:-}  # containers to stop, space-separated
stopped=""
restore() {
    for c in $stopped; do docker start "$c" >/dev/null 2>&1 || say "could not start $c"; done
    sleep 20
    for c in $stopped; do say "started $c: $(docker ps --filter "name=^/$c\$" --format "{{.Status}}")"; done
    stopped=""
}
trap restore EXIT
trap "exit 130" INT TERM HUP
say "queued; receipts $OUT; waiting for the current GPU phase"
while tmux has-session -t speed-all-gpu 2>/dev/null; do sleep 20; done
for c in $QUIET; do
    state=$(docker inspect -f "{{.State.Status}}" "$c" 2>/dev/null)
    if [ "$state" = running ]; then docker stop "$c" >/dev/null && stopped="$stopped $c"; fi
    say "$c: was $state, now $(docker inspect -f "{{.State.Status}}" "$c" 2>/dev/null)"
done
sleep 120
say "settled; load $(cut -d" " -f1-3 /proc/loadavg); top: $(ps -eo pcpu,comm --sort=-pcpu | sed -n 2,5p | tr -s " " | paste -sd";")"
env DICT_SYSTEMS="udlex-rs udlex-py spacy-sm spacy-lg simplemma lemmagen3 udpipe1 corenlp" \
    NEURAL_SYSTEMS="stanza trankit spacy-trf" "$H/all_languages.sh" "$OUT" 2>&1 | tee -a "$OUT.log"
say "CPU phases done; GPU phase"
OPTS=$("$S/.venv-udlex/bin/python" -I -c "import json,sys; print(\" \".join(json.load(open(sys.argv[1]))[\"options\"]))" "$S/receipts/auto-20261119T081436Z/sweep/chosen.json")
(
    exec 9>"$LOCK"; flock 9
    UDLEX_SPEED_DICTS=$REL taskset -c 16-31 "$S/.venv-udlex/bin/python" -I "$H/bench.py" run "$S" "$OUT" \
        --langs $(cat "$OUT/languages.txt") --rounds 3 --windows 3 --cpu1 4 $OPTS --stanza-gpu-tokens 0 \
        --systems stanza trankit spacy-trf --modes gpu \
        --note "C9 quiet run, GPU phase; vLLM and the UDLEX_SPEED_PAUSE services stopped, see services.log" \
        >>"$OUT/run.log" 2>&1 || say "GPU phase: bench.py exited with $?"
)
UDLEX_SPEED_DICTS=$REL "$S/.venv-udlex/bin/python" -I "$H/bench.py" summary "$OUT" > "$OUT/summary.log" 2>&1 || say "no summary"
say "QUIET-RUN-DONE $OUT/SUMMARY.md"
restore
