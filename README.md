# embed-images

Image thumbnailing + TensorRT embedding extraction with duplicate detection. Includes a web UI for browsing results.

## Build

```bash
docker build -t embed-images .
```

## Web UI

```bash
docker run --gpus all -p 8000:8000 -v /path/to/data:/workspace embed-images
```

Open http://localhost:8000 — configure the pipeline, hit Run, and browse duplicate pairs side-by-side.

## CLI

```bash
docker run --gpus all -v /path/to/data:/workspace embed-images cli \
  --model /workspace/clip-vit-b32-visual.onnx \
  --images-dir /workspace/images/ \
  --output-dir /workspace/output/ \
  --find-duplicates \
  --top-k 50
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *required* | Path to ONNX embedding model (e.g. CLIP ViT-B/32) |
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
