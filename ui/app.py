"""Web UI for embed-images pipeline."""

import asyncio
import json
import os
import subprocess
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="embed-images")

BINARY = os.environ.get("EMBED_IMAGES_BIN", "/usr/local/bin/embed-images")
WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))


class PipelineConfig(BaseModel):
    model: str
    images_dir: str
    output_dir: str = "/workspace/output"
    thumbnail_size: int = 224
    batch_size: int = 32
    output_format: str = "npz"
    find_duplicates: bool = True
    top_k: int = 50
    duplicate_threshold: float = 0.9


# Serve thumbnails and output files
app.mount("/files", StaticFiles(directory="/workspace", check_dir=False), name="files")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/results")
async def get_results():
    """Load the latest pipeline results."""
    output_dir = WORKSPACE / "output"

    result = {"duplicates": [], "paths": [], "status": "no_results"}

    dupes_path = output_dir / "duplicates.json"
    if dupes_path.exists():
        with open(dupes_path) as f:
            result["duplicates"] = json.load(f)

    paths_path = output_dir / "paths.json"
    if paths_path.exists():
        with open(paths_path) as f:
            result["paths"] = json.load(f)

    if result["duplicates"] or result["paths"]:
        result["status"] = "complete"

    return result


@app.websocket("/ws/run")
async def run_pipeline(ws: WebSocket):
    """Run the pipeline, streaming log output over WebSocket."""
    await ws.accept()

    try:
        data = await ws.receive_json()
        config = PipelineConfig(**data)

        cmd = [
            BINARY,
            "--model", config.model,
            "--images-dir", config.images_dir,
            "--output-dir", config.output_dir,
            "--thumbnail-size", str(config.thumbnail_size),
            "--batch-size", str(config.batch_size),
            "--output-format", config.output_format,
        ]
        if config.find_duplicates:
            cmd += [
                "--find-duplicates",
                "--top-k", str(config.top_k),
                "--duplicate-threshold", str(config.duplicate_threshold),
            ]

        await ws.send_json({"type": "status", "message": "Starting pipeline..."})
        await ws.send_json({"type": "cmd", "message": " ".join(cmd)})

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async for line in proc.stdout:
            text = line.decode().rstrip()
            if text:
                await ws.send_json({"type": "log", "message": text})

        code = await proc.wait()

        if code == 0:
            # Load results
            dupes_path = Path(config.output_dir) / "duplicates.json"
            duplicates = []
            if dupes_path.exists():
                with open(dupes_path) as f:
                    duplicates = json.load(f)

            await ws.send_json({
                "type": "complete",
                "message": f"Pipeline finished successfully",
                "duplicates": duplicates,
            })
        else:
            await ws.send_json({
                "type": "error",
                "message": f"Pipeline failed with exit code {code}",
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
