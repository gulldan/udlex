#!/usr/bin/env bash
set -euo pipefail

plugin_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
os_home=$(cd -- "${1:?Usage: test.sh OPENSEARCH_HOME BUNDLE [LANGUAGE TOKEN EXPECTED_LEMMA]}" && pwd)
bundle=$(cd -- "${2:?Pass the udlex bundle directory}" && pwd)
jdk_dir=${OPENSEARCH_JAVA_HOME:-$os_home/jdk}
shopt -s nullglob
core_jars=("$os_home"/lib/opensearch-[0-9]*.jar)
[[ ${#core_jars[@]} == 1 ]] || { echo 'Expected one OpenSearch distribution' >&2; exit 1; }
version=${core_jars[0]##*/opensearch-}
archive="$plugin_dir/dist/analysis-udlex-${version%.jar}.zip"
if [[ $(uname -s) != Linux || ! -f "$archive" ]]; then
    archive="$plugin_dir/dist/analysis-udlex-${version%.jar}-$(uname -m).zip"
fi
archive=${PLUGIN_ZIP:-"$archive"}
[[ -f "$archive" ]] || { echo "Build $archive first" >&2; exit 1; }
archive=$(cd -- "$(dirname -- "$archive")" && pwd)/$(basename -- "$archive")

work=$(mktemp -d "${TMPDIR:-/tmp}/udlex-contract.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
(cd -- "$work" && "$jdk_dir/bin/jar" --extract --file "$archive")
jars=("$work"/analysis-udlex-*.jar)
[[ ${#jars[@]} == 1 ]] || { echo 'Expected exactly one plugin JAR in the ZIP' >&2; exit 1; }
mkdir "$work/classes"
test_sources="$plugin_dir/src/test/java"
classpath="${jars[0]}:$os_home/lib/*"
"$jdk_dir/bin/javac" -encoding UTF-8 -Xlint:all -Werror -cp "$classpath" --source-path "$test_sources" \
    -d "$work/classes" "$test_sources/org/udlex/opensearch/NativeContractTest.java"

# Pass ASCII argv even when an SSH session uses the C locale.
encode_argument() {
    printf '%s' "$1" | base64 | tr -d '\r\n'
}
"$jdk_dir/bin/java" -ea -Xcheck:jni --enable-native-access=ALL-UNNAMED \
    -cp "$work/classes:$classpath" org.udlex.opensearch.NativeContractTest \
    "$(encode_argument "$bundle")" "$(encode_argument "${3:-rus}")" \
    "$(encode_argument "${4:-котами}")" "$(encode_argument "${5:-кот}")"
