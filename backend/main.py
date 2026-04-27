"""FastAPI backend for the plagiarism detection system."""

from __future__ import annotations

import json
import logging
import shutil
import sys
import uuid
from pathlib import Path
from typing import Optional

# Ensure backend/ is on sys.path for sibling imports
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
try:
    from sse_starlette.sse import EventSourceResponse
    _HAS_SSE = True
except ImportError:
    _HAS_SSE = False

from schemas import (
    DetectRequest,
    DetectionResults,
    JobInfo,
    SystemConfig,
    UploadResponse,
)
from runner import run_detection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Plagiarism Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ─────────────────────────────────────
# Production: use Redis/DB. For this project, a dict is fine.
_jobs: dict[str, dict] = {}

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def _api_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "api_config.json"


def _load_api_config() -> dict | None:
    p = _api_config_path()
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ── Endpoints ───────────────────────────────────────────────

@app.get("/api/config", response_model=SystemConfig)
async def get_config():
    """Return system configuration info (API availability, etc.)."""
    cfg = _load_api_config()
    if cfg is None:
        # Check env vars
        import os
        api_key = os.environ.get("MODELSCOPE_API_KEY")
        if api_key:
            return SystemConfig(
                api_available=True,
                api_provider="openai",
                model_name=os.environ.get("MODELSCOPE_MODEL", "deepseek-ai/DeepSeek-V3.1"),
            )
        return SystemConfig(api_available=False)
    provider_cfg = cfg.get("modelscope", cfg.get("openai", cfg))
    return SystemConfig(
        api_available=True,
        api_provider=provider_cfg.get("provider", "openai"),
        model_name=provider_cfg.get("model", ""),
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(
    files: list[UploadFile] = File(...),
    mode: str = Form("all"),
    target_names: str = Form(""),
):
    """Upload text files and create a detection job."""
    job_id = uuid.uuid4().hex[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    saved = []
    for f in files:
        if not f.filename:
            continue
        dest = job_dir / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)

    if not saved:
        raise HTTPException(400, "No files uploaded")

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "uploaded",
        "files": saved,
        "mode": mode,
        "target_names": [n.strip() for n in target_names.split(",") if n.strip()],
        "results": None,
    }

    return UploadResponse(job_id=job_id, files=saved)


@app.post("/api/detect")
async def detect(req: DetectRequest):
    """Run plagiarism detection synchronously."""
    if req.job_id not in _jobs:
        raise HTTPException(404, f"Job {req.job_id} not found")

    job = _jobs[req.job_id]
    job_dir = UPLOAD_DIR / req.job_id

    if not job_dir.exists():
        raise HTTPException(404, "Job files not found")

    job["status"] = "detecting"

    try:
        results = run_detection(
            job_dir,
            sensitivity=req.sensitivity,
            enable_ai=req.enable_ai,
            agent_depth=req.agent_depth,
            enable_crosslingual=req.enable_crosslingual,
            detection_mode=req.detection_mode,
            target_names=req.target_names or job.get("target_names", []),
        )
        job["status"] = "done"
        job["results"] = results

        # Generate report files
        _generate_reports(job_dir, results)

        return results
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        logger.exception("Detection failed")
        raise HTTPException(500, str(e))


@app.post("/api/detect/stream")
async def detect_stream(req: DetectRequest):
    """Run detection — alias for synchronous detect (SSE optional)."""
    return await detect(req)


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get detection results for a completed job."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")

    job = _jobs[job_id]
    if job["status"] == "detecting":
        return {"status": "detecting"}
    if job["status"] == "error":
        return {"status": "error", "error": job.get("error")}
    if job["results"] is None:
        return {"status": "pending"}

    return job["results"]


@app.get("/api/results/{job_id}/export")
async def export_report(job_id: str, format: str = "json"):
    """Export detection report in various formats."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")

    job_dir = UPLOAD_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, "Job directory not found")

    format = format.lower()
    file_map = {
        "csv": job_dir / "pair_summary.csv",
        "json": job_dir / "pair_results.json",
        "docx": job_dir / "plagiarism_report.docx",
        "docx_summary": job_dir / "plagiarism_summary_report.docx",
        "para_csv": job_dir / "paragraph_summary.csv",
        "para_json": job_dir / "paragraph_results.json",
        "para_docx": job_dir / "plagiarism_paragraph_report.docx",
    }

    if format not in file_map:
        raise HTTPException(400, f"Unsupported format: {format}. Use: {list(file_map.keys())}")

    filepath = file_map[format]
    if not filepath.exists():
        raise HTTPException(404, f"Report file not found: {filepath.name}")

    media_types = {
        "csv": "text/csv",
        "json": "application/json",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "docx_summary": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "para_csv": "text/csv",
        "para_json": "application/json",
        "para_docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    return FileResponse(
        filepath,
        media_type=media_types.get(format, "application/octet-stream"),
        filename=filepath.name,
    )


@app.get("/api/jobs/{job_id}", response_model=JobInfo)
async def get_job_info(job_id: str):
    """Get job status information."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    job = _jobs[job_id]
    return JobInfo(
        job_id=job["job_id"],
        status=job["status"],
        files=job["files"],
        error=job.get("error"),
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files."""
    if job_id in _jobs:
        del _jobs[job_id]
    job_dir = UPLOAD_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    return {"status": "deleted"}


# ── Helpers ─────────────────────────────────────────────────

def _generate_reports(job_dir: Path, results: dict):
    """Generate report files using the existing reporting module."""
    try:
        from plagiarism_checker.pipeline import PipelineConfig
        from plagiarism_checker.reporting import (
            write_summary_csv,
            write_pair_results,
            write_word_report,
            write_word_summary_report,
            write_paragraph_summary,
        )

        sent_stats = results.get("sent_stats", [])
        sent_details = results.get("sent_details", [])
        para_stats = results.get("para_stats", [])
        para_details = results.get("para_details", [])

        if sent_stats:
            write_summary_csv(job_dir / "pair_summary.csv", sent_stats)
        if sent_details:
            write_pair_results(job_dir / "pair_results.json", sent_details)
        if sent_stats and sent_details:
            try:
                write_word_report(job_dir / "plagiarism_report.docx", sent_stats, sent_details)
                write_word_summary_report(job_dir / "plagiarism_summary_report.docx", sent_stats)
            except Exception as e:
                logger.warning("Word report generation failed: %s", e)
        if para_stats:
            write_paragraph_summary(job_dir / "paragraph_summary.csv", para_stats)
        if para_details:
            write_pair_results(job_dir / "paragraph_results.json", para_details)
            try:
                para_with_hits = []
                for d in para_details:
                    c = dict(d)
                    c["hits"] = d.get("matches", [])
                    para_with_hits.append(c)
                write_word_report(job_dir / "plagiarism_paragraph_report.docx", para_stats, para_with_hits)
            except Exception as e:
                logger.warning("Paragraph Word report failed: %s", e)

        logger.info("Reports generated in %s", job_dir)
    except Exception as e:
        logger.warning("Report generation failed: %s", e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
