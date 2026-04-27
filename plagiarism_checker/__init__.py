"""Plagiarism checker package."""

from .pipeline import PlagiarismPipeline
from .cli import main

__all__ = ["PlagiarismPipeline", "main"]
