"""Minimal web app for running the annotation pipeline on custom text."""

import asyncio
import logging
from pathlib import Path
import os
import sys
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rl_detector.annotate import annotate_with_runtime, create_runtime
from rl_detector.config import CFG

logger = logging.getLogger(__name__)


def _configure_pipeline_logging() -> None:
    # route rl_detector logs to uvicorn's console handlers
    uvicorn_err = logging.getLogger("uvicorn.error")
    pipeline_logger = logging.getLogger("rl_detector")
    pipeline_logger.setLevel(logging.INFO)
    attached = 0
    if uvicorn_err.handlers:
        for h in uvicorn_err.handlers:
            if h not in pipeline_logger.handlers:
                pipeline_logger.addHandler(h)
                attached += 1
    # fallback for environments where uvicorn handlers are not available yet
    if not pipeline_logger.handlers:
        fallback = logging.StreamHandler(sys.stdout)
        fallback.setLevel(logging.INFO)
        fallback.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        pipeline_logger.addHandler(fallback)
        attached += 1
    pipeline_logger.propagate = False
    pipeline_logger.info("webui | pipeline logging configured (handlers=%d)", len(pipeline_logger.handlers))
    logger.info("webui | app logger active; attached pipeline handlers=%d", attached)


class _JobLogHandler(logging.Handler):
    def __init__(self, store: dict):
        super().__init__(level=logging.INFO)
        self.store = store
        self.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))

    def emit(self, record):
        try:
            msg = self.format(record)
            self.store["logs"].append(msg)
        except Exception:
            return


_jobs: dict[str, dict] = {}
_result_cache: dict[str, dict] = {}   # text -> finished result, for slow-response fallback
_startup_runtime: dict | None = None
_startup_checkpoint: str | None = None
_engine_status: str = "not_configured"
_engine_error: str | None = None


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)


class AnalyzeStartResponse(BaseModel):
    job_id: str


class AnalyzeStatusResponse(BaseModel):
    status: str
    progress_pct: int
    progress_stage: str
    logs: list[str]
    next_offset: int
    result: dict | None = None
    error: str | None = None


def _cfg_checkpoint() -> str | None:
    # keep config lookup tiny and explicit
    web_cfg = getattr(CFG, "web", None)
    if web_cfg is None:
        return None
    return getattr(web_cfg, "checkpoint_path", None)


def _segments_from_indicators(text: str, indicators: list[dict]) -> list[dict]:
    # deterministic span matching to keep UI highlights stable
    items = sorted(indicators, key=lambda x: text.find(x["span_text"]))
    result = []
    offset = 0
    for ind in items:
        span = ind["span_text"]
        pos = text.find(span, offset)
        if pos == -1:
            continue
        if pos > offset:
            result.append({"type": "plain", "text": text[offset:pos]})
        result.append(
            {
                "type": "tell",
                "text": span,
                "explanation": ind.get("explanation", ""),
                "score": ind.get("frozen_score", 0.0),
            }
        )
        offset = pos + len(span)
    if offset < len(text):
        result.append({"type": "plain", "text": text[offset:]})
    return result


_CACHE_TIMEOUT_S = 10


async def _run_job(job_id: str, text: str, checkpoint: str) -> None:
    job = _jobs[job_id]
    handler = _JobLogHandler(job)
    pipeline_logger = logging.getLogger("rl_detector")
    pipeline_logger.setLevel(logging.INFO)
    pipeline_logger.addHandler(handler)
    try:
        job["status"] = "running"
        job["progress_pct"] = 5
        job["progress_stage"] = "Starting pipeline"
        job["logs"].append("webui | job started")

        def on_progress(pct: int, stage: str) -> None:
            job["progress_pct"] = max(0, min(100, int(pct)))
            job["progress_stage"] = stage

        job["logs"].append("webui | using preloaded startup sampler")
        use_cache = getattr(getattr(CFG, "web", None), "result_cache", True)
        inference_task = asyncio.create_task(
            annotate_with_runtime(text, _startup_runtime, progress_cb=on_progress)
        )
        if use_cache:
            done, _ = await asyncio.wait({inference_task}, timeout=_CACHE_TIMEOUT_S)
            if not done:
                cached = _result_cache.get(text)
                if cached is not None:
                    inference_task.cancel()
                    job["logs"].append(f"webui | inference timed out after {_CACHE_TIMEOUT_S}s, returning cached result")
                    result = cached
                else:
                    job["logs"].append(f"webui | inference took >{_CACHE_TIMEOUT_S}s, no cache — waiting for completion")
                    result = await inference_task
            else:
                result = inference_task.result()
        else:
            result = await inference_task

        result["segments"] = _segments_from_indicators(text, result["indicators"])
        result["used_checkpoint"] = checkpoint
        if use_cache:
            _result_cache[text] = result
        job["result"] = result
        job["status"] = "done"
        job["progress_pct"] = 100
        job["progress_stage"] = "Complete"
        job["logs"].append("webui | job finished")
    except Exception as e:
        job["status"] = "error"
        job["progress_stage"] = "Failed"
        job["error"] = str(e)
        job["logs"].append(f"webui | job failed: {e}")
    finally:
        pipeline_logger.removeHandler(handler)


app = FastAPI(title="RL Detector Web UI", version="0.1.0")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def _startup() -> None:
    global _engine_status
    global _engine_error
    global _startup_checkpoint
    _configure_pipeline_logging()
    _startup_checkpoint = os.getenv("RL_DETECTOR_CHECKPOINT") or _cfg_checkpoint()
    if _startup_checkpoint:
        _engine_status = "loading"
        logger.info("webui | scheduling startup sampler preload for checkpoint: %s", _startup_checkpoint)
        asyncio.create_task(_warmup_runtime())
    else:
        _engine_status = "not_configured"
        logger.info("webui | no startup checkpoint configured")


async def _warmup_runtime() -> None:
    global _startup_runtime
    global _engine_status
    global _engine_error
    global _startup_checkpoint
    try:
        logger.info("webui | startup sampler preload started")
        _startup_runtime = await create_runtime(checkpoint_path=_startup_checkpoint)
        _engine_status = "ready"
        _engine_error = None
        logger.info("webui | startup sampler ready")
    except Exception as e:
        _startup_runtime = None
        _engine_status = "error"
        _engine_error = str(e)
        logger.exception("webui | startup sampler preload failed")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/config")
async def api_config() -> dict:
    checkpoint = os.getenv("RL_DETECTOR_CHECKPOINT") or _cfg_checkpoint()
    return {
        "checkpoint_configured": bool(checkpoint),
        "checkpoint_path": checkpoint,
        "startup_sampler_ready": _startup_runtime is not None,
        "engine_status": _engine_status,
        "engine_error": _engine_error,
        "base_model": CFG.model.base_model,
    }


@app.post("/api/analyze")
async def api_analyze(payload: AnalyzeRequest) -> dict:
    if _startup_runtime is None or _startup_checkpoint is None:
        detail = "Inference engine is not ready yet."
        if _engine_status == "loading":
            detail = "Inference engine is loading; please wait a bit and retry."
        if _engine_status == "error":
            detail = f"Inference engine failed to load: {_engine_error}"
        if _engine_status == "not_configured":
            detail = "No startup checkpoint/runtime available. Set RL_DETECTOR_CHECKPOINT or config.yaml web.checkpoint_path and restart the app."
        raise HTTPException(
            status_code=503,
            detail=detail,
        )
    result = await annotate_with_runtime(payload.text, _startup_runtime)
    result["segments"] = _segments_from_indicators(payload.text, result["indicators"])
    result["used_checkpoint"] = _startup_checkpoint
    return result


@app.post("/api/analyze/start", response_model=AnalyzeStartResponse)
async def api_analyze_start(payload: AnalyzeRequest) -> AnalyzeStartResponse:
    if _startup_runtime is None or _startup_checkpoint is None:
        detail = "Inference engine is not ready yet."
        if _engine_status == "loading":
            detail = "Inference engine is loading; please wait a bit and retry."
        if _engine_status == "error":
            detail = f"Inference engine failed to load: {_engine_error}"
        if _engine_status == "not_configured":
            detail = "No startup checkpoint/runtime available. Set RL_DETECTOR_CHECKPOINT or config.yaml web.checkpoint_path and restart the app."
        raise HTTPException(
            status_code=503,
            detail=detail,
        )
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "queued",
        "progress_pct": 0,
        "progress_stage": "Queued",
        "logs": [],
        "result": None,
        "error": None,
    }
    asyncio.create_task(_run_job(job_id, payload.text, _startup_checkpoint))
    return AnalyzeStartResponse(job_id=job_id)


@app.get("/api/analyze/status/{job_id}", response_model=AnalyzeStatusResponse)
async def api_analyze_status(job_id: str, after: int = 0) -> AnalyzeStatusResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    logs = job["logs"]
    start = max(0, after)
    chunk = logs[start:]
    return AnalyzeStatusResponse(
        status=job["status"],
        progress_pct=job.get("progress_pct", 0),
        progress_stage=job.get("progress_stage", "Queued"),
        logs=chunk,
        next_offset=len(logs),
        result=job["result"],
        error=job["error"],
    )
