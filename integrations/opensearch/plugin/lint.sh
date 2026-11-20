#!/usr/bin/env bash
set -euo pipefail
plugin_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$plugin_dir/../../.." && pwd)
if [[ $# == 0 ]]; then
    exec docker build --target checks --file "$plugin_dir/Dockerfile" "$repo_dir"
fi
os_home=$(cd -- "$1" && pwd)
jdk=${OPENSEARCH_JAVA_HOME:-$os_home/jdk}
work=$(mktemp -d "${TMPDIR:-/tmp}/udlex-plugin-lint.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT

# CHECKSTYLE_JAR also supports offline builds. Verify both sources identically.
version=14.4.0
expected=c0a521e54eabd1ece4ff7bd84cf231b012615745ed3aa4fb27ab0ca73097250c
jar=${CHECKSTYLE_JAR:-"$work/checkstyle.jar"}
if [[ -z ${CHECKSTYLE_JAR:-} ]]; then
    curl --fail --silent --show-error --location --retry 3 --proto '=https' --proto-redir '=https' \
        "https://github.com/checkstyle/checkstyle/releases/download/checkstyle-$version/checkstyle-$version-all.jar" \
        --output "$jar"
fi
if command -v sha256sum >/dev/null; then
    actual=$(sha256sum < "$jar")
else
    actual=$(shasum -a 256 < "$jar")
fi
[[ ${actual%% *} == "$expected" ]] || { echo 'Checkstyle JAR checksum mismatch' >&2; exit 1; }

"$jdk/bin/java" -jar "$jar" -c "$plugin_dir/checks/checkstyle.xml" "$plugin_dir/src"
shellcheck --severity=style "$plugin_dir"/*.sh
"$jdk/bin/javac" -encoding UTF-8 -Xlint:all -Werror -h "$work" -d "$work" \
    "$plugin_dir/src/main/java/org/udlex/opensearch/UdlexNative.java"
case "$(uname -s)" in
    Linux) platform=linux ;;
    Darwin) platform=darwin ;;
    *) echo 'Unsupported JNI platform' >&2; exit 1 ;;
esac
"${CLANG_TIDY:-clang-tidy}" --config-file="$plugin_dir/checks/clang-tidy.yml" \
    "$plugin_dir/src/main/cpp/udlex_jni.cpp" -- -std=c++17 \
    -isystem "$jdk/include" -isystem "$jdk/include/$platform" \
    -I"$work" -I"$repo_dir/rust/capi/include"
