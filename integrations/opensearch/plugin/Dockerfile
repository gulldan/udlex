FROM opensearchproject/opensearch:3.10.0@sha256:02d6d6e40c820ae08233b60540aa5d3d9364d64e2dc724e38d6c8c794879b2a0 AS distribution
FROM rust:1.100.0-bookworm@sha256:ade39a710ef85f3506cffcfec3e8a24ba090badef1dff2fb798f71d878833fea AS compiler
ENV RUSTUP_TOOLCHAIN=1.100.0

# The host compiler produces one library and the cross compiler produces the other.
RUN case "$(dpkg --print-architecture)" in \
        amd64) peer=g++-aarch64-linux-gnu; target=aarch64-unknown-linux-gnu ;; \
        arm64) peer=g++-x86-64-linux-gnu; target=x86_64-unknown-linux-gnu ;; \
        *) exit 1 ;; \
    esac && apt-get update && apt-get install -y --no-install-recommends "$peer" && \
    rm -rf /var/lib/apt/lists/* && rustup target add "$target"

WORKDIR /build
COPY rust-toolchain.toml .
COPY rust/ rust/
COPY integrations/opensearch/plugin/build.sh integrations/opensearch/plugin/build.sh
COPY integrations/opensearch/plugin/src/main/ integrations/opensearch/plugin/src/main/
RUN --mount=from=distribution,source=/usr/share/opensearch,target=/distribution,ro \
    --mount=type=cache,target=/usr/local/cargo/registry \
    find rust rust-toolchain.toml integrations/opensearch/plugin -type f -print0 | \
        LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1 > SOURCE_SHA256 && \
    bash integrations/opensearch/plugin/build.sh /distribution 3.10.0 --both-linux

FROM compiler AS checks
RUN apt-get update && apt-get install -y --no-install-recommends clang-tidy shellcheck && \
    rm -rf /var/lib/apt/lists/* && rustup component add clippy rustfmt
COPY integrations/opensearch/plugin/lint.sh integrations/opensearch/plugin/test.sh integrations/opensearch/plugin/
COPY integrations/opensearch/plugin/checks/ integrations/opensearch/plugin/checks/
COPY integrations/opensearch/plugin/src/test/ integrations/opensearch/plugin/src/test/
RUN --mount=from=distribution,source=/usr/share/opensearch,target=/distribution,ro \
    --mount=type=cache,target=/usr/local/cargo/registry \
    cargo fmt --manifest-path rust/Cargo.toml --all -- --check && \
    cargo clippy --locked --manifest-path rust/Cargo.toml -p udlex-capi --all-targets -- -D warnings && \
    bash integrations/opensearch/plugin/lint.sh /distribution && \
    bash integrations/opensearch/plugin/test.sh /distribution integrations/opensearch/plugin/src/test/dictionaries eng ox longexamplelemma

FROM scratch
COPY --from=compiler /build/integrations/opensearch/plugin/dist/analysis-udlex-3.10.0.zip /
