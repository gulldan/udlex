ARG OPENSEARCH_BASE=opensearchproject/opensearch:3.10.0@sha256:02d6d6e40c820ae08233b60540aa5d3d9364d64e2dc724e38d6c8c794879b2a0
FROM ${OPENSEARCH_BASE}

RUN --mount=type=bind,target=/tmp/udlex-kit \
    (cd /tmp/udlex-kit/plugins && sha256sum -c SHA256SUMS) && \
    /usr/share/opensearch/bin/opensearch-plugin install --batch \
    file:///tmp/udlex-kit/plugins/analysis-udlex-3.10.0.zip && \
    python3 /tmp/udlex-kit/install-dictionaries.py /usr/share/opensearch/config/udlex-224-20261120

COPY 90-udlex.options /usr/share/opensearch/config/jvm.options.d/90-udlex.options
