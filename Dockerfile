FROM nvcr.io/nvidia/tensorrt:25.12-py3 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    git \
    libopencv-dev \
    libz-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY CMakeLists.txt .
COPY include/ include/
COPY src/ src/

RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORRT_ROOT=/usr \
    && cmake --build build --parallel "$(nproc)"

# ── Runtime stage ────────────────────────────────────────────────
FROM nvcr.io/nvidia/tensorrt:25.12-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core-dev \
    libopencv-imgproc-dev \
    libopencv-imgcodecs-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/build/embed-images /usr/local/bin/embed-images

WORKDIR /workspace

ENTRYPOINT ["embed-images"]
