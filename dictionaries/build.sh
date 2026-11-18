#!/bin/sh
# Build every dictionary package from the pinned public inputs, then score it.
# Usage: dictionaries/build.sh DIR [DIR_WITH_1.0.0_DICTIONARIES]
# Needs Python >= 3.11 with udlex 1.0.0 built from this checkout and pymorphy3 2.0.6.
set -eu
dir=$1
here=$(cd "$(dirname "$0")" && pwd)
py=${PYTHON:-python3}
ud=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["ud"]["sha256"])' "$here/sources.json")
mkdir -p "$dir/cache" "$dir/work"
"$py" "$here/fetch.py" "$dir/cache"
"$py" "$here/splits.py" "$dir/cache/$ud" "$dir/work"
"$py" "$here/extract.py" "$dir/cache" "$dir/work"
"$py" "$here/choose.py" "$dir/work"
"$py" "$here/package.py" "$dir/work" "$dir/release"
if [ $# -gt 1 ]; then
    "$py" "$here/evaluate.py" "$dir/work" "$dir/release" "$2"
    "$py" "$here/report.py" "$dir/release"
fi
