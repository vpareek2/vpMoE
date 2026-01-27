ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/vpmoe

COPY requirements.txt /workspace/vpmoe/
RUN python3 -m pip install --break-system-packages --upgrade pip && \
    python3 -m pip install --break-system-packages --no-cache-dir -r requirements.txt

ARG TRANSFORMERS_REPO=https://github.com/huggingface/transformers.git
ARG TRANSFORMERS_REF=v4.57.6
RUN git clone --filter=blob:none --depth 1 --branch "${TRANSFORMERS_REF}" "${TRANSFORMERS_REPO}" /opt/transformers && \
    python3 -m pip uninstall -y transformers || true && \
    python3 -m pip install --break-system-packages --no-cache-dir -e /opt/transformers

# This image is a pinned dependency/runtime environment. The repo is bind-mounted
# at runtime (see docker/compose.yml), so do not COPY the full workspace here.
#
# Keep repo-root and src/ on the import path, and expose DistillKit from the
# bind-mounted checkout (so local edits are live).
ENV PYTHONPATH=/workspace/vpmoe:/workspace/vpmoe/src:/workspace/vpmoe/src/DistillKit

CMD ["/bin/bash"]
