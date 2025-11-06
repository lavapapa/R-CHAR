"""
R-CHAR: Role-Consistent Hierarchical Adaptive Reasoning

A metacognition-driven framework for enhancing role-playing performance in large language models.
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from .core.core_engine import RCharengine
from .core.trajectory_synthesis import TrajectorySynthesis
from .evaluation.social_evaluator import SocialBenchEvaluator

__all__ = [
    "RCharengine",
    "TrajectorySynthesis",
    "SocialBenchEvaluator"
]