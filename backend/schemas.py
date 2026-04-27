"""Pydantic request/response models for the plagiarism detection API."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel


# ── Request models ──────────────────────────────────────────

class DetectRequest(BaseModel):
    job_id: str
    sensitivity: Literal["low", "medium", "high"] = "medium"
    enable_ai: bool = False
    agent_depth: Literal["quick", "thorough"] = "quick"
    enable_crosslingual: bool = False
    detection_mode: Literal["target", "all"] = "all"
    target_names: list[str] = []


# ── Response models ─────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id: str
    files: list[str]


class SystemConfig(BaseModel):
    api_available: bool
    api_provider: str = ""
    model_name: str = ""


class DetectionResults(BaseModel):
    sent_stats: list = []
    sent_details: list = []
    para_stats: list = []
    para_details: list = []
    agent_reports: list = []
    auto_adjusted: bool = False
    original_threshold: Optional[float] = None
    adjusted_threshold: Optional[float] = None


class JobInfo(BaseModel):
    job_id: str
    status: Literal["uploaded", "detecting", "done", "error"]
    files: list[str] = []
    error: Optional[str] = None
