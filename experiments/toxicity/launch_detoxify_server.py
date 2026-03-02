#!/usr/bin/env python3
"""
Launch a detoxify toxicity classifier server on GPU.

Exposes a /classify endpoint that accepts batches of text and returns
toxicity scores using the detoxify library (BERT/RoBERTa-based classifiers).

Usage:
    python launch_detoxify_server.py --port 8001
    python launch_detoxify_server.py --port 8001 --model-type unbiased --device cuda:1

Endpoint contract:
    POST /classify  {"texts": ["text1", "text2", ...]}
    -> 200          {"scores": {"toxicity": [0.95, 0.1], "severe_toxicity": [0.2, 0.01], ...}}

    POST /classify  {"texts": ["single text"]}
    -> 200          {"scores": {"toxicity": [0.95], ...}}
"""

import os
import argparse
import signal
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    import uvloop  # noqa: F401

    _LOOP = "uvloop"
except ImportError:
    _LOOP = "asyncio"

# ── Pydantic schemas ─────────────────────────────────────────────────────


class ClassifyRequest(BaseModel):
    texts: list[str] = Field(..., description="List of texts to classify")


# ── App & state ──────────────────────────────────────────────────────────

app = FastAPI()
_classifier = None
_executor = ThreadPoolExecutor(max_workers=1)

# ── Inference ────────────────────────────────────────────────────────────


def _classify_sync(texts: list[str]) -> dict[str, list[float]]:
    assert _classifier is not None
    results = _classifier.predict(texts)
    # detoxify returns float for single input, list for batch — normalise to list
    return {k: v if isinstance(v, list) else [v] for k, v in results.items()}


# ── Routes ───────────────────────────────────────────────────────────────


@app.post("/classify")
async def classify(request: ClassifyRequest):
    if not request.texts:
        return JSONResponse(status_code=400, content={"error": "texts cannot be empty"})
    import asyncio

    loop = asyncio.get_running_loop()
    scores = await loop.run_in_executor(_executor, _classify_sync, request.texts)
    return {"scores": scores}


@app.get("/health")
async def health():
    return JSONResponse(status_code=200, content={"status": "ok"})


# ── Model loading ────────────────────────────────────────────────────────


def load_classifier(args: argparse.Namespace) -> None:
    global _classifier

    # Add detoxify to path so we can import it from the cloned repo
    sys.path.insert(
        0,
        str(__import__("pathlib").Path(__file__).resolve().parent.parent / "detoxify"),
    )

    from detoxify import Detoxify

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading detoxify model: {args.model_type}")
    print(f"  device: {device}")

    _classifier = Detoxify(args.model_type, device=device)

    # Warmup
    print("  Running warmup inference...")
    _ = _classifier.predict(["warmup text"])
    print("  Warmup complete.")
    print(f"  Classes: {_classifier.class_names}")
    print("Classifier ready.\n")

    os.environ["BEAVER_CLASSIFIER_ADDR"] = f"http://localhost:{args.port}"


# ── CLI ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detoxify classifier server for BEAVER toxicity verification.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8001, help="TCP port")
    parser.add_argument(
        "--model-type",
        type=str,
        default="unbiased",
        choices=[
            "original",
            "unbiased",
            "multilingual",
            "original-small",
            "unbiased-small",
        ],
        help="Detoxify model type (default: unbiased — RoBERTa-based, includes sexual_explicit)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to load model on (auto, cpu, cuda, cuda:0, etc.)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    def _sig(signum, frame):
        print("\nShutting down classifier server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    load_classifier(args)

    print(f"Starting classifier server at http://{args.host}:{args.port}")
    print(
        f"Set BEAVER_CLASSIFIER_ADDR=http://{args.host}:{args.port} to connect BEAVER to this server."
    )
    print("Press Ctrl+C to stop.\n")
    uvicorn.run(app, host=args.host, port=args.port, loop=_LOOP)


if __name__ == "__main__":
    main()
