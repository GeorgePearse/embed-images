#!/usr/bin/env python3
"""Export a timm model to ONNX for use with embed-images TensorRT pipeline.

Usage:
    # Default: ConvNeXt-Tiny with DINOv3 pretraining (768-dim, good general-purpose embeddings)
    python scripts/export_timm_model.py

    # Any timm model:
    python scripts/export_timm_model.py --model resnet50.a1_in1k --output model.onnx

    # List available pretrained models matching a pattern:
    python scripts/export_timm_model.py --list "convnext*"

Requirements:
    pip install timm torch
"""

import argparse
import sys

import timm
import torch


def list_models(pattern: str) -> None:
    models = timm.list_models(pattern, pretrained=True)
    if not models:
        print(f"No pretrained models matching '{pattern}'")
        return
    for m in models:
        print(m)


def export(model_name: str, output_path: str, size: int, opset: int) -> None:
    print(f"Loading {model_name} (pretrained=True, num_classes=0)...")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()

    dummy = torch.randn(1, 3, size, size)
    with torch.no_grad():
        out = model(dummy)
    embed_dim = out.shape[-1]
    print(f"Embedding dim: {embed_dim}")

    print(f"Exporting to {output_path} (opset {opset})...")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    print(f"Done. {output_path} ready for embed-images --model {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a timm model to ONNX for TensorRT embedding extraction"
    )
    parser.add_argument(
        "--model",
        default="convnext_tiny.dinov3_lvd1689m",
        help="timm model name (default: convnext_tiny.dinov3_lvd1689m)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output ONNX path (default: <model_name>.onnx)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Input resolution (default: 224)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13, good TensorRT compat)",
    )
    parser.add_argument(
        "--list",
        metavar="PATTERN",
        default=None,
        help="List available pretrained models matching pattern, then exit",
    )
    args = parser.parse_args()

    if args.list is not None:
        list_models(args.list)
        sys.exit(0)

    output = args.output or f"{args.model.replace('.', '_').replace('/', '_')}.onnx"
    export(args.model, output, args.size, args.opset)


if __name__ == "__main__":
    main()
