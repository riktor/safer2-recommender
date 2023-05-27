FROM python:3.8.16-slim-bullseye
LABEL maintainer="anonymous"

RUN apt update && apt install -y build-essential curl git gcc g++ libpthread-stubs0-dev && \
    curl -LO "https://github.com/bazelbuild/bazelisk/releases/download/v1.1.0/bazelisk-linux-amd64" && \
    mkdir -p "/usr/bin/" && \
    mv bazelisk-linux-amd64 "/usr/bin/bazel" && \
    chmod +x "/usr/bin/bazel"

WORKDIR /frecsys
ADD codes.tar .

RUN bazel build run_model && ln -s $(bazel info bazel-bin) bazel-bin

RUN pip3 install numpy pandas && mkdir dataset && \
    python scripts/generate_data.py
