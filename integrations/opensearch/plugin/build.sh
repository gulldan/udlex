#!/usr/bin/env bash
set -euo pipefail

plugin_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$plugin_dir/../../.." && pwd)
dist="$plugin_dir/dist"
mkdir -p "$dist"
work=$(mktemp -d "$dist/.build.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
if [[ $# == 0 ]]; then
    # The public entry point uses only Docker. Publish after a successful export.
    (cd -- "$work" && docker build --file "$plugin_dir/Dockerfile" --output type=local,dest=. "$repo_dir")
    mv -f -- "$work/analysis-udlex-3.10.0.zip" "$dist/"
    echo "Built $dist/analysis-udlex-3.10.0.zip"
    exit 0
fi
os_home=$(cd -- "$1" && pwd)
jdk_dir=${OPENSEARCH_JAVA_HOME:-$os_home/jdk}
shopt -s nullglob
core_jars=("$os_home"/lib/opensearch-[0-9]*.jar)
if [[ ${#core_jars[@]} != 1 || ! -x "$jdk_dir/bin/javac" ]]; then
    echo 'Supply an OpenSearch distribution with lib/opensearch-VERSION.jar and a full JDK.' >&2
    exit 1
fi
version=${core_jars[0]##*/opensearch-}
version=${version%.jar}
if [[ $# -ge 2 && "$2" != "$version" ]]; then
    echo "Requested OpenSearch $2 but the distribution is $version" >&2
    exit 1
fi

jdk_version=$("$jdk_dir/bin/javac" -version)
jdk_major=${jdk_version#javac }
jdk_major=${jdk_major%%[.+-]*}
[[ $jdk_major =~ ^[0-9]+$ ]] || { echo "Unexpected javac version: $jdk_version" >&2; exit 1; }

mkdir -p "$work/classes" "$work/headers" "$work/package/native"
sources=()
while IFS= read -r -d '' source; do sources+=("$source"); done \
    < <(find "$plugin_dir/src/main/java" -name '*.java' -type f -print0)
"$jdk_dir/bin/javac" --release "$jdk_major" -encoding UTF-8 -Xlint:all -Werror \
    -cp "$os_home/lib/*" -h "$work/headers" -d "$work/classes" "${sources[@]}"

export CARGO_TARGET_DIR="$plugin_dir/build/rust-target"
export RUSTFLAGS="${RUSTFLAGS:-} -C relocation-model=pic"
case "$(uname -s)" in
    Linux)
        jni_platform=linux
        native_name=libudlex_jni.so
        native_flags=(-shared "-Wl,-z,defs" "-Wl,-z,relro" "-Wl,-z,now" "-Wl,--exclude-libs,ALL")
        native_libraries=(-pthread -ldl -lm -lrt -lutil)
        ;;
    Darwin)
        jni_platform=darwin
        native_name=libudlex_jni.dylib
        native_flags=(-dynamiclib)
        native_libraries=(-framework Security -framework CoreFoundation)
        ;;
    *) echo 'Unsupported native build platform' >&2; exit 1 ;;
esac
compile_native() {
    local cpu=$1 cxx=$2 target=$3
    local package_dir="$work/package/native/$cpu"
    mkdir -p "$package_dir"
    (cd -- "$repo_dir" && cargo build --locked --release --manifest-path rust/Cargo.toml \
        -p udlex-capi --target "$target")
    "$cxx" -std=c++17 -O3 -fPIC -fvisibility=hidden \
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror \
        -isystem "$jdk_dir/include" -isystem "$jdk_dir/include/$jni_platform" \
        -I"$work/headers" -I"$repo_dir/rust/capi/include" "${native_flags[@]}" \
        "$plugin_dir/src/main/cpp/udlex_jni.cpp" "$CARGO_TARGET_DIR/$target/release/libudlex_capi.a" \
        "${native_libraries[@]}" -o "$package_dir/$native_name"
}

if [[ ${3:-} == --both-linux ]]; then
    export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
    for cpu in x86_64 aarch64; do
        compile_native "$cpu" "$cpu-linux-gnu-g++" "$cpu-unknown-linux-gnu"
    done
    platform=linux/amd64,linux/arm64
    zip_path="$dist/analysis-udlex-$version.zip"
else
    cpu=$(uname -m)
    [[ $cpu != arm64 ]] || cpu=aarch64
    target=$(rustc -vV | sed -n 's/^host: //p')
    compile_native "$cpu" "${CXX:-c++}" "$target"
    platform="$(uname -s)-$(uname -m)"
    zip_path="$dist/analysis-udlex-$version-$(uname -m).zip"
fi

plugin_version=0.1.1
archive_date=2000-01-01T00:00:00Z
"$jdk_dir/bin/jar" --create --date="$archive_date" \
    --file "$work/package/analysis-udlex-$plugin_version.jar" -C "$work/classes" .
cp "$plugin_dir/src/main/resources/plugin-security.policy" "$work/package/"
cat > "$work/package/plugin-descriptor.properties" <<EOF
description=Native udlex dictionary lemmatization token filter
version=$plugin_version
name=analysis-udlex
classname=org.udlex.opensearch.UdlexPlugin
java.version=$jdk_major
opensearch.version=$version
has.native.controller=false
EOF
cat > "$work/package/build-info.txt" <<EOF
opensearch_version=$version
java_version=$jdk_version
platform=$platform
rust_version=$(rustc --version)
EOF
if [[ -f "$repo_dir/SOURCE_SHA256" ]]; then
    printf 'source_sha256=%s\n' "$(cat "$repo_dir/SOURCE_SHA256")" >> "$work/package/build-info.txt"
fi
"$jdk_dir/bin/jar" --create --no-manifest --date="$archive_date" \
    --file "$work/plugin.zip" -C "$work/package" .
mv -f -- "$work/plugin.zip" "$zip_path"
echo "Built $zip_path"
