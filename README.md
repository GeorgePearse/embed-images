# embed-images

Image thumbnailing + TensorRT embedding extraction with duplicate detection. Includes a web UI for browsing results.

## Build

```bash
docker build -t embed-images .
```

### Build from source (without Docker)

```bash
mkdir build && cd build
cmake .. \
  -DTENSORRT_ROOT=/usr/targets/x86_64-linux-gnu \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10
make -j$(nproc)
```

Adjust `CUDA_ARCHITECTURES` for your GPU (75=T4, 80=A100, 86=RTX3090, 89=L4, 90=H100).

#### Dependencies

- CUDA toolkit
- TensorRT (with ONNX parser)
- OpenCV (`libopencv-dev`)
- CMake >= 3.18

Header-only deps (fetched automatically): CLI11, nlohmann/json, spdlog, cnpy.

## Web UI

```bash
docker run --gpus all -p 8000:8000 -v /path/to/data:/workspace embed-images
```

Open http://localhost:8000 — configure the pipeline, hit Run, and browse duplicate pairs side-by-side.

## Getting an ONNX model

Any image classification or embedding model exported to ONNX works. The easiest path is [pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models):

```bash
pip install timm torch

# Default: ConvNeXt-Tiny with DINOv3 pretraining (768-dim embeddings, good for duplicates)
python scripts/export_timm_model.py

# Other models:
python scripts/export_timm_model.py --model resnet50.a1_in1k
python scripts/export_timm_model.py --model efficientnet_b0.ra4_e3600_r224_in1k

# Browse available models:
python scripts/export_timm_model.py --list "convnext*"
python scripts/export_timm_model.py --list "efficientnet*"
```

The export script outputs an ONNX file with dynamic batch axes and opset 13 (good TensorRT compatibility). The first time you run `embed-images` with a new model, TensorRT builds and caches an optimized FP16 engine — subsequent runs load instantly.

### Recommended models

| Model | Dim | Speed | Quality | Notes |
|-------|-----|-------|---------|-------|
| `convnext_tiny.dinov3_lvd1689m` | 768 | Fast | Best | DINOv3 self-supervised, great for similarity |
| `convnext_small.dinov3_lvd1689m` | 768 | Medium | Better | Larger DINOv3 variant |
| `resnet50.a1_in1k` | 2048 | Fast | Good | ImageNet supervised baseline |
| `efficientnet_b0.ra4_e3600_r224_in1k` | 1280 | Fastest | Good | Smallest, good enough for exact dupes |

## CLI

```bash
# Via Docker:
docker run --gpus all -v /path/to/data:/workspace embed-images cli \
  --model /workspace/clip-vit-b32-visual.onnx \
  --images-dir /workspace/images/ \
  --output-dir /workspace/output/ \
  --find-duplicates \
  --top-k 50

# Or directly:
embed-images \
  --model convnext_tiny_dinov3_lvd1689m.onnx \
  --images-dir /path/to/images \
  --output-dir /path/to/output \
  --thumbnail-size 224 \
  --batch-size 32 \
  --output-format npz \
  --find-duplicates \
  --top-k 100 \
  --duplicate-threshold 0.95
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *required* | Path to ONNX embedding model |
| `--images-dir` | *required* | Directory containing input images (recursive) |
| `--output-dir` | *required* | Output directory for thumbnails + embeddings |
| `--thumbnail-size` | 224 | Resize images to NxN |
| `--batch-size` | 32 | Inference batch size |
| `--output-format` | npz | `npz` or `json` |
| `--find-duplicates` | off | Find most similar image pairs |
| `--top-k` | 50 | Number of duplicate pairs to report |
| `--duplicate-threshold` | 0.9 | Minimum cosine similarity |

### Output

- `output/thumbnails/` — resized PNGs
- `output/embeddings.npz` — `(N, embed_dim)` float32 array
- `output/paths.json` — ordered list of original image paths
- `output/duplicates.json` — top-K similar pairs (if `--find-duplicates`)
- `output/engine_cache/` — cached TensorRT engine (reused on subsequent runs)

### Choosing a threshold

| Threshold | What it catches |
|-----------|-----------------|
| >= 0.9999 | Byte-identical images (different filenames) |
| >= 0.99 | Visually identical (same scene, compression differences) |
| >= 0.95 | Near-duplicates (same scene, minor temporal changes) |
| >= 0.90 | Similar scenes (same camera angle, different content) |

## Example: finding duplicates in a dataset

```bash
# 1. Export a pretrained model
python scripts/export_timm_model.py

# 2. Run embedding + duplicate detection
embed-images \
  --model convnext_tiny_dinov3_lvd1689m.onnx \
  --images-dir ~/data/images \
  --output-dir ~/data/embeddings \
  --find-duplicates \
  --top-k 100 \
  --duplicate-threshold 0.95

# 3. Review duplicates.json
cat ~/data/embeddings/duplicates.json | python -m json.tool | head -30
```

On a T4 GPU with 9,544 images: ~45s for thumbnailing, ~45s for inference, ~55s for pairwise similarity scan.
