#!/usr/bin/env bash
# C9 on every language with a UD 2.19 test text and a 1.0.0 dictionary (169), CPU modes.
# Neural systems are timed on one core only: their 16-thread cells stall whenever another
# process takes a core (final of 19.11). GPU cells go in a separate phase with vLLM off.
#   all_languages.sh OUT          (server, in tmux) prepare, wait for the stand, time, summarise
# The stand as in run.sh; UDLEX_SPEED_RELEASE, if set, is a rebuilt release timed instead of
# the build's own (the build still gives the languages and the Python of prepare).
set -uo pipefail
RUN=${UDLEX_SPEED_RUN:?set UDLEX_SPEED_RUN to the run directory}
H=$RUN/harness/speed
C=${UDLEX_SPEED_COMPARE:?set UDLEX_SPEED_COMPARE to the quality comparison}
D=${UDLEX_SPEED_DICTS:?set UDLEX_SPEED_DICTS to the dictionary build}
export UDLEX_SPEED_DICTS=${UDLEX_SPEED_RELEASE:-$D}
PY=$RUN/.venv-udlex/bin/python
LOCK=${UDLEX_SPEED_LOCK:?set UDLEX_SPEED_LOCK to the exclusive lock file of the stand}
OUT=$1
# Systems to time, and a file whose presence pauses timing before each cell (a build).
DICT_SYSTEMS=${DICT_SYSTEMS:-udlex-rs udlex-py spacy-sm spacy-lg simplemma lemmagen3 udpipe1 corenlp}
NEURAL_SYSTEMS=${NEURAL_SYSTEMS:-stanza trankit spacy-trf}
PAUSE_FILE=${PAUSE_FILE:-}
say() { echo "$(TZ=Asia/Tokyo date '+%F %T %Z') $*"; }
mkdir -p "$OUT"
LANGS=$("$PY" -I - "$C" "$D" <<'EOF'
import json, os, sys
C = f"{sys.argv[1]}/census/inputs/order"
inv = json.load(open(f"{C}/inventory.json"))["languages"]
ev = {r["iso639_3"] for r in json.load(open(f"{sys.argv[2]}/release/evaluation.json"))["languages"]}
print(" ".join(sorted(i for i in inv if os.path.exists(f"{C}/{i}/test.jsonl") and i in ev)))
EOF
)
echo "$LANGS" > "$OUT/languages.txt"
say "languages: $(wc -w <<<"$LANGS")"
cd "$RUN"
for m in $("$PY" -I -c 'import json,sys; c=json.load(open(sys.argv[1]))["cells"]; print(" ".join(c[i]["model"] for i in sys.argv[2:] if c.get(i,{}).get("status")=="scored"))' "$C/results-census/udpipe1.json" $LANGS); do
    [[ -s udpipe1-models/$m.udpipe ]] ||
        curl -fsSL -o "udpipe1-models/$m.udpipe" "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/$m.udpipe" ||
        say "udpipe1 model $m: download failed"
done
say "udpipe1 models: $(ls udpipe1-models | wc -l)"
taskset -c 0-7 "$D/.venv/bin/python" -I "$H/bench.py" prepare "$RUN" $LANGS > "$OUT/prepare.log" 2>&1 ||
    { say "prepare failed, see prepare.log"; exit 1; }
say "prepared"
while tmux ls 2>/dev/null | grep -qE '^(udlex-exphybrid-|encoding-)'; do sleep 60; done
OPTS=$("$PY" -I -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["options"]))' "$RUN/receipts/auto-20261119T081436Z/sweep/chosen.json")
say "waiting for the lock; options $OPTS"
exec 9>"$LOCK"; flock 9
say "lock taken; load $(cut -d' ' -f1-3 /proc/loadavg)"
pause=(); [[ -n $PAUSE_FILE ]] && pause=(--pause-while "$PAUSE_FILE")
common=(--langs $LANGS --rounds 3 --windows 3 --cpu1 4 $OPTS "${pause[@]}"
    --note "C9 on all 169 languages, CPU modes; vLLM and other services as vllm.log or services.log record; udlex $(cat "$RUN/src-udlex/COMMIT")")
taskset -c 16-31 "$PY" -I "$H/bench.py" run "$RUN" "$OUT" "${common[@]}" \
    --systems $DICT_SYSTEMS \
    --modes 1c thr16 proc16 >>"$OUT/run.log" 2>&1 || say "dictionary systems: bench.py exited with $?"
say "dictionary systems done"
[[ -z $NEURAL_SYSTEMS ]] ||
    taskset -c 16-31 "$PY" -I "$H/bench.py" run "$RUN" "$OUT" "${common[@]}" \
        --systems $NEURAL_SYSTEMS --modes 1c >>"$OUT/run.log" 2>&1 ||
    say "neural systems: bench.py exited with $?"
say "neural systems (one core) done"
"$PY" -I "$H/bench.py" summary "$OUT" > "$OUT/summary.log" 2>&1 || say "no summary"
say "ALL-LANGUAGES-DONE $OUT/SUMMARY.md"
