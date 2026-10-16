"""Core package for AI-Examples: deterministic micro-generators, metrics, and benchmarks."""

from core.env import EnvInfo, setup_env
from core import data
from core import metrics
from core import benchmark

__all__ = ["EnvInfo", "setup_env", "data", "metrics", "benchmark"]
