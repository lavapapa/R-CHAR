"""
Evaluation module for R-CHAR framework.

This module contains evaluation tools and benchmarks for assessing
role-playing performance, including SocialBench integration and
custom evaluation metrics.
"""

from .social_evaluator import SocialBenchEvaluator
from .roleplay_evaluator import RoleplayEvaluator
from .metrics import EvaluationMetrics, MetricCalculator

__all__ = [
    "SocialBenchEvaluator",
    "RoleplayEvaluator",
    "EvaluationMetrics",
    "MetricCalculator"
]