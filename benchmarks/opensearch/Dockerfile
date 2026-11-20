FROM opensearchproject/opensearch:3.10.0@sha256:02d6d6e40c820ae08233b60540aa5d3d9364d64e2dc724e38d6c8c794879b2a0 AS opensearch
FROM rust:1.100.0-bookworm@sha256:ade39a710ef85f3506cffcfec3e8a24ba090badef1dff2fb798f71d878833fea AS compiler
ENV RUSTUP_TOOLCHAIN=1.100.0 CARGO_NET_OFFLINE=true
WORKDIR /build
COPY build-inputs.sha256 .
COPY build-inputs/ files/
RUN cd files && sha256sum -c ../build-inputs.sha256
RUN dpkg -i files/zip_*_$(dpkg --print-architecture).deb && \
    mkdir source stock /artifacts && \
    tar -xzf files/udlex-opensearch-source.tar.gz -C source --strip-components=1 && \
    tar -xzf files/udlex-core-01addf4.tar.gz -C stock --strip-components=1 && \
    tar -xzf files/cargo-vendor.tar.gz && \
    printf '%s\n' '[source.crates-io]' 'replace-with = "vendored"' '[source.vendored]' \
      'directory = "/build/vendor"' > "$CARGO_HOME/config.toml" && \
    echo 0a3334f519e2ee0dda7865f0eca7882833a04267 > source/SOURCE_COMMIT
RUN --mount=from=opensearch,source=/usr/share/opensearch,target=/distribution,ro \
    bash source/integrations/opensearch/plugin/build.sh /distribution 3.10.0 && \
    cp source/integrations/opensearch/plugin/dist/*.zip /artifacts/analysis-udlex.zip && \
    cargo build --offline --locked --release --manifest-path stock/rust/Cargo.toml -p udlex-capi && \
    cp stock/rust/target/release/libudlex_capi.so /artifacts/ && \
    cp files/analysis-*-3.10.0.zip /artifacts/

FROM python:3.14-slim-bookworm@sha256:3fe1a0b6c2d6b2ebb572906cdabeca75aeb558c0c17db5ffefbcf8094871c32b
ENV LANG=C.UTF-8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY --from=opensearch /usr/share/opensearch /opt/opensearch
COPY --from=compiler /artifacts/ /opt/bench/artifacts/
RUN set -eu; mkdir -p /opt/optional /results /opt/opensearch/logs; \
    for archive in /opt/bench/artifacts/*.zip; do \
      /opt/opensearch/bin/opensearch-plugin install --batch "file://$archive"; \
    done; \
    for plugin in analysis-udlex analysis-icu analysis-kuromoji analysis-smartcn; do \
      mv "/opt/opensearch/plugins/$plugin" /opt/optional/; \
    done; \
    chmod -R a+rX /opt/optional /opt/opensearch/config; \
    chmod 0777 /opt/opensearch/plugins /opt/opensearch/logs; \
    useradd --create-home --uid 1000 bench; \
    python -c 'from compression import zstd'
COPY runner/ /opt/bench/
COPY reference/ /opt/bench/reference/
COPY benchmark.lock.json build-inputs.sha256 /opt/bench/
ENTRYPOINT ["python", "/opt/bench/run.py"]
CMD ["full"]
