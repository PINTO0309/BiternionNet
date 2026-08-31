"""TownCentre synthetic-head generation and validation pipeline."""

from .generate import PipelineError, build_plan, create_plan

__all__ = ["PipelineError", "build_plan", "create_plan"]
