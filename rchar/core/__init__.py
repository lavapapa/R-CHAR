"""
Core module for R-CHAR framework.

This module contains the main optimization algorithms and utilities
for role-playing enhancement through thinking trajectory guidance.
"""

from .core_engine import RCharengine, ScenarioResult, RoleplayResult, EvaluateResult
from .trajectory_synthesis import TrajectorySynthesis
from .prompts import *

__all__ = [
    "RCharengine",
    "TrajectorySynthesis",
    "ScenarioResult",
    "RoleplayResult",
    "EvaluateResult"
]