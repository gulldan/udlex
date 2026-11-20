#!/usr/bin/env bash
set -euo pipefail
plugin_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
os_home=${1:?Usage: test.sh OPENSEARCH_HOME BUNDLE [LANGUAGE TOKEN EXPECTED_LEMMA]}
bundle=${2:?Pass the real udlex bundle directory}
language=${3:-rus}
token=${4:-котами}
expected=${5:-кот}
os_home=$(cd -- "$os_home" && pwd)
jdk_dir=${OPENSEARCH_JAVA_HOME:-$os_home/jdk}
package_dir="$plugin_dir/build/package"
test_classes="$plugin_dir/build/test-classes"
if [[ ! -f "$package_dir/analysis-udlex-0.1.0.jar" ]]; then
    echo "Run build.sh first" >&2
    exit 1
fi
mkdir -p "$test_classes"
find "$plugin_dir/src/test/java" -name '*.java' -print | LC_ALL=C sort > "$plugin_dir/build/test-sources.txt"
"$jdk_dir/bin/javac" -encoding UTF-8 -Xlint:all \
    -cp "$package_dir/analysis-udlex-0.1.0.jar:$os_home/lib/*" \
    -d "$test_classes" @"$plugin_dir/build/test-sources.txt"
# Under SSH's C locale, the JVM launcher may decode each UTF-8 byte in argv as
# an ASCII replacement character. Pass ASCII base64 and decode explicitly in
# Java so the test verifies JNI rather than the launcher's locale heuristics.
encode_argument() {
    printf '%s' "$1" | base64 | tr -d '\r\n'
}
"$jdk_dir/bin/java" -ea -Xcheck:jni --enable-native-access=ALL-UNNAMED \
    -cp "$test_classes:$package_dir/analysis-udlex-0.1.0.jar:$os_home/lib/*" \
    org.udlex.opensearch.NativeContractTest \
    "$(encode_argument "$bundle")" "$(encode_argument "$language")" \
    "$(encode_argument "$token")" "$(encode_argument "$expected")"
