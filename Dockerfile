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

# Python is already in the TensorRT image
RUN pip install --no-cache-dir fastapi uvicorn[standard] websockets

COPY --from=builder /build/build/embed-images /usr/local/bin/embed-images
COPY ui/ /opt/embed-images/ui/

WORKDIR /workspace

EXPOSE 8000

# Default: run the web UI. Override with CLI args to use the binary directly.
# Examples:
#   docker run --gpus all -p 8000:8000 -v ./data:/workspace embed-images
#   docker run --gpus all -v ./data:/workspace embed-images cli --model ...
COPY <<'ENTRYPOINT' /opt/embed-images/entrypoint.sh
#!/bin/bash
set -e

if [ "$1" = "cli" ]; then
    shift
    exec embed-images "$@"
else
    exec uvicorn app:app --host 0.0.0.0 --port 8000 --app-dir /opt/embed-images/ui "$@"
fi
ENTRYPOINT

RUN chmod +x /opt/embed-images/entrypoint.sh
ENTRYPOINT ["/opt/embed-images/entrypoint.sh"]
