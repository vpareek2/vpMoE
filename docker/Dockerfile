ARG BASE_IMAGE=nvcr.io/nvidia/vllm:25.12.post1-py3
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/vpmoe

COPY requirements.txt /workspace/vpmoe/
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY . /workspace/vpmoe

ENV PYTHONPATH=/workspace/vpmoe:/workspace/vpmoe/vpmoe/Megatron-vpmoe

CMD ["/bin/bash"]
