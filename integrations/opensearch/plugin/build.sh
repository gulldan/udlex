#!/usr/bin/env bash
set -euo pipefail

# Build against the exact OpenSearch distribution that will load the plugin.
# No Gradle/Maven dependency resolution or downloaded OpenSearch jars are used.
plugin_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$plugin_dir/../../.." && pwd)
os_home=${1:?Usage: build.sh /path/to/opensearch [exact-version]}
os_home=$(cd -- "$os_home" && pwd)
jdk_dir=${OPENSEARCH_JAVA_HOME:-$os_home/jdk}
if [[ ! -x "$jdk_dir/bin/javac" ]]; then
    echo "No javac in $jdk_dir; set OPENSEARCH_JAVA_HOME to the distribution's full JDK" >&2
    exit 1
fi

shopt -s nullglob
core_jars=("$os_home"/lib/opensearch-[0-9]*.jar)
if [[ ${#core_jars[@]} != 1 ]]; then
    echo "Expected exactly one opensearch-VERSION.jar under $os_home/lib" >&2
    exit 1
fi
version=${core_jars[0]##*/opensearch-}
version=${version%.jar}
if [[ $# -ge 2 && "$2" != "$version" ]]; then
    echo "Requested OpenSearch $2 but supplied distribution is $version" >&2
    exit 1
fi
jdk_version=$("$jdk_dir/bin/javac" -version 2>&1)
jdk_major=${jdk_version#javac }
jdk_major=${jdk_major%%[.+-]*}
if [[ ! "$jdk_major" =~ ^[0-9]+$ ]]; then
    echo "Cannot read javac version: $jdk_version" >&2
    exit 1
fi

build_dir="$plugin_dir/build"
dist_dir="$plugin_dir/dist"
stage_dir="$build_dir/package"
mkdir -p "$build_dir" "$dist_dir"
rm -rf -- "$build_dir/classes" "$build_dir/headers" "$stage_dir"
mkdir -p "$build_dir/classes" "$build_dir/headers" "$stage_dir/native"

export CARGO_TARGET_DIR="$build_dir/rust-target"
export RUSTFLAGS="${RUSTFLAGS:-} -C relocation-model=pic"
(cd -- "$repo_dir" && cargo build --locked --release --manifest-path rust/Cargo.toml -p udlex-capi)
capi_archive="$CARGO_TARGET_DIR/release/libudlex_capi.a"
if [[ ! -f "$capi_archive" ]]; then
    echo "Missing C ABI static archive: $capi_archive" >&2
    exit 1
fi

case "$(uname -s)" in
    Linux)
        native_name=libudlex_jni.so
        native_flags=(-shared -Wl,-z,defs -Wl,-z,relro -Wl,-z,now -Wl,--exclude-libs,ALL)
        native_libraries=(-pthread -ldl -lm -lrt -lutil)
        jni_platform=linux
        ;;
    Darwin)
        native_name=libudlex_jni.dylib
        native_flags=(-dynamiclib)
        native_libraries=(-framework Security -framework CoreFoundation)
        jni_platform=darwin
        ;;
    *) echo "Unsupported native build platform" >&2; exit 1 ;;
esac

"${CXX:-c++}" -std=c++17 -O3 -fPIC -fvisibility=hidden -Wall -Wextra -Werror \
    -I"$jdk_dir/include" -I"$jdk_dir/include/$jni_platform" -I"$repo_dir/rust/capi/include" \
    "${native_flags[@]}" "$plugin_dir/src/main/cpp/udlex_jni.cpp" "$capi_archive" \
    "${native_libraries[@]}" -o "$stage_dir/native/$native_name"

find "$plugin_dir/src/main/java" -name '*.java' -print | LC_ALL=C sort > "$build_dir/java-sources.txt"
"$jdk_dir/bin/javac" --release "$jdk_major" -encoding UTF-8 -Xlint:all \
    -cp "$os_home/lib/*" -h "$build_dir/headers" -d "$build_dir/classes" @"$build_dir/java-sources.txt"
"$jdk_dir/bin/jar" --create --file "$stage_dir/analysis-udlex-0.1.0.jar" -C "$build_dir/classes" .
cp "$plugin_dir/src/main/resources/plugin-security.policy" "$stage_dir/"
cat > "$stage_dir/plugin-descriptor.properties" <<EOF
description=Native udlex dictionary lemmatization token filter
version=0.1.0
name=analysis-udlex
classname=org.udlex.opensearch.UdlexPlugin
java.version=$jdk_major
opensearch.version=$version
has.native.controller=false
EOF
cat > "$stage_dir/build-info.txt" <<EOF
opensearch_version=$version
java_version=$jdk_version
platform=$(uname -s)-$(uname -m)
rust_version=$(rustc --version 2>/dev/null || true)
repository_commit=$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || cat "$repo_dir/SOURCE_COMMIT" 2>/dev/null || true)
EOF

zip_path="$dist_dir/analysis-udlex-$version-$(uname -m).zip"
rm -f -- "$zip_path"
(cd -- "$stage_dir" && zip -qr "$zip_path" ./*)
echo "Built $zip_path"
echo "Contract test: $plugin_dir/test.sh $os_home /path/to/bundle rus котами кот"
