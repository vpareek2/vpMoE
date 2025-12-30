ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN uv venv /opt/venv --python python3 --system-site-packages
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace/vpmoe

COPY pyproject.toml uv.lock /workspace/vpmoe/
RUN uv sync --frozen --no-install-project

COPY . /workspace/vpmoe

ENV PYTHONPATH=/workspace/vpmoe:/workspace/vpmoe/vpmoe/Megatron-LM

CMD ["/bin/bash"]
