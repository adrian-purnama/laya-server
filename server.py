"""
HTTP server for Laya.

Install:
    pip install fastapi "uvicorn[standard]"

Run (recommended - controls host/port itself, banner always matches):
    python server.py
    # custom port:  $env:PORT=9000; python server.py   (PowerShell)

Run (alternative, uvicorn CLI):
    uvicorn server:app --host 0.0.0.0 --port 8000
"""
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

# Laya's own docs note this avoids a TensorFlow-probe deadlock in transformers.
os.environ.setdefault("USE_TF", "0")

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from laya import Router

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loaded once at process startup, kept resident for the life of the server.
router = Router(preload=True, device=DEVICE)

_WARMUP_STATE = {"body": "warmup ping"}
_WARMUP_QUESTIONS = {
    "check": {"type": "noul", "instructions": "Is this a warmup request?"}
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("Laya Decision Server")
    print(f"  Listening on : http://{HOST}:{PORT}")
    print(f"  Device       : {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")

    # One health check: run a real forward pass now, at startup, so a broken
    # checkpoint/config fails loudly here instead of on someone's first request.
    try:
        t0 = time.perf_counter()
        router.predict(_WARMUP_STATE, _WARMUP_QUESTIONS)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  Health check : OK ({elapsed_ms:.1f} ms)")
    except Exception as e:
        print(f"  Health check : FAILED - {e}")
    print("=" * 50)

    yield  # server runs here

    print("Laya Decision Server shutting down.")


app = FastAPI(title="Laya Decision Server", lifespan=lifespan)


class PredictRequest(BaseModel):
    state: Dict[str, Any]
    questions: Dict[str, Any]
    model: Optional[str] = None  # e.g. "typed-decisions" to force a checkpoint


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.model:
        result = router.predict(req.state, req.questions, model=req.model)
    else:
        result = router.predict(req.state, req.questions)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)