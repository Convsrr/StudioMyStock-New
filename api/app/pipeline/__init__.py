"""Image processing pipeline. Each stage is a small, testable function."""

from .orchestrator import PipelineParams, PipelineResult, run_pipeline

__all__ = ["PipelineParams", "PipelineResult", "run_pipeline"]
