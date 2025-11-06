"""
Trajectory Synthesis Module

This module implements alternative methods for thinking trajectory optimization
used in ablation studies to validate the effectiveness of R-CHAR components.
"""

from typing import NamedTuple, List, Dict, Any, Optional, Tuple
import asyncio
import json
import os
from openai import AsyncClient
from tenacity import retry, stop_after_attempt

from .core_engine import RCharengine, ScenarioResult, RoleplayResult, EvaluateResult, LLMConfig
from .utils.llms import ask, ask_json, ask_messages
from .utils.response_decoders import extract_tags_dict, extract_key_values
from .prompts import (
    REFERENCE_OPTIMIZATION_PROMPT, SIMPLE_SELECTION_PROMPT,
    REWRITE_OPTIMIZATION_PROMPT
)


class TrajectorySynthesis:
    """
    Alternative trajectory synthesis methods for ablation studies

    Provides different approaches to thinking trajectory optimization
    for validating the effectiveness of R-CHAR framework components.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, debug_mode: bool = False):
        """
        Initialize trajectory synthesis module

        Args:
            config: Configuration dictionary
            debug_mode: Enable debug logging
        """
        self.config = config or {}
        self.debug_mode = debug_mode

    def debug_print(self, *args, **kwargs):
        """Print debug information if debug mode is enabled"""
        if self.debug_mode:
            print("🔍 DEBUG:", *args, **kwargs)

    @retry(stop=stop_after_attempt(3))
    async def reference_optimization(self, character: str, scenario: str, instruction: str,
                                   current_answer: str, reference_answer: str,
                                   llm_client: AsyncClient, model: str) -> str:
        """
        Reference-based optimization method

        Uses a high-quality reference answer to guide improvement
        of the current response.

        Args:
            character: Character description
            scenario: Scenario context
            instruction: Role-playing instruction
            current_answer: Current response to improve
            reference_answer: High-quality reference response
            llm_client: LLM client instance
            model: Model name

        Returns:
            Improved response
        """
        self.debug_print(f"🔗 Running reference optimization...")

        prompt = REFERENCE_OPTIMIZATION_PROMPT.format(
            character=character,
            scenario=scenario,
            instruction=instruction,
            answer=current_answer,
            reference_answer=reference_answer
        )

        response = await ask(prompt, model, llm_client)
        result = extract_tags_dict(response)

        improved_answer = result.get('improved_response', current_answer)
        explanation = result.get('improvement_explanation', '')

        self.debug_print(f"✅ Reference optimization completed")
        self.debug_print(f"💡 Explanation: {explanation[:100]}...")

        return improved_answer

    @retry(stop=stop_after_attempt(3))
    async def simple_selection(self, character: str, scenario: str, instruction: str,
                              response_options: List[str], llm_client: AsyncClient,
                              model: str) -> Tuple[str, str]:
        """
        Simple selection method

        Directly selects the best response from multiple options
        without using predefined evaluation criteria.

        Args:
            character: Character description
            scenario: Scenario context
            instruction: Role-playing instruction
            response_options: List of response options
            llm_client: LLM client instance
            model: Model name

        Returns:
            Tuple of (best_response, selection_reasoning)
        """
        self.debug_print(f"🎯 Running simple selection...")

        # Format response options
        options_text = "\n\n".join([
            f"Option {chr(65 + i)}:\n{opt}"
            for i, opt in enumerate(response_options)
        ])

        prompt = SIMPLE_SELECTION_PROMPT.format(
            character=character,
            scenario=scenario,
            instruction=instruction,
            responses=options_text
        )

        response = await ask(prompt, model, llm_client)
        result = extract_tags_dict(response)

        best_response = result.get('best_response', response_options[0])
        reasoning = result.get('selection_reasoning', '')
        ranking = result.get('response_ranking', '')

        self.debug_print(f"✅ Simple selection completed")
        self.debug_print(f"🧠 Reasoning: {reasoning[:100]}...")

        return best_response, reasoning

    @retry(stop=stop_after_attempt(3))
    async def rewrite_optimization(self, character: str, scenario: str, instruction: str,
                                  think: str, answer: str, llm_client: AsyncClient,
                                  model: str) -> Tuple[str, str]:
        """
        Rewrite optimization method

        Rewrites the thinking trajectory to improve clarity, consistency,
        and depth while maintaining the original response intent.

        Args:
            character: Character description
            scenario: Scenario context
            instruction: Role-playing instruction
            think: Original thinking process
            answer: Original response
            llm_client: LLM client instance
            model: Model name

        Returns:
            Tuple of (rewritten_think, rewritten_answer)
        """
        self.debug_print(f"✏️ Running rewrite optimization...")

        prompt = REWRITE_OPTIMIZATION_PROMPT.format(
            character=character,
            scenario=scenario,
            instruction=instruction,
            think=think,
            answer=answer
        )

        response = await ask(prompt, model, llm_client)
        result = extract_tags_dict(response)

        rewritten_think = result.get('rewritten_thinking', think)
        rewritten_answer = result.get('rewritten_response', answer)
        explanation = result.get('optimization_explanation', '')

        self.debug_print(f"✅ Rewrite optimization completed")
        self.debug_print(f"💡 Explanation: {explanation[:100]}...")

        return rewritten_think, rewritten_answer

    async def reference_optimization_loop(self, character: str, scenario: str, instruction: str,
                                        reference_answers: List[str], llm_config: LLMConfig,
                                        max_iterations: int = 3) -> Tuple[str, float, List[Dict]]:
        """
        Reference-based optimization loop

        Iteratively improves responses using reference answers.

        Args:
            character: Character description
            scenario: Scenario context
            instruction: Role-playing instruction
            reference_answers: List of high-quality reference answers
            llm_config: LLM configuration
            max_iterations: Maximum optimization iterations

        Returns:
            Tuple of (best_answer, best_score, optimization_history)
        """
        self.debug_print(f"🔄 Starting reference optimization loop...")

        # Initialize with a basic response
        base_engine = RCharengine(debug_mode=self.debug_mode)
        initial_result = await base_engine.execute_roleplay(
            character, scenario, instruction,
            llm_client=llm_config.execute_roleplay_client,
            model=llm_config.execute_roleplay_model
        )

        current_answer = initial_result.answer
        best_answer = current_answer
        best_score = 0
        history = []

        for iteration in range(max_iterations):
            self.debug_print(f"📍 Reference iteration {iteration + 1}/{max_iterations}")

            # Select a random reference answer
            import random
            reference = random.choice(reference_answers)

            # Optimize using reference
            improved_answer = await self.reference_optimization(
                character, scenario, instruction,
                current_answer, reference,
                llm_config.evaluate_performance_client,
                llm_config.evaluate_performance_model
            )

            # Evaluate the improved answer
            eval_result = await base_engine.evaluate_performance(
                character, scenario, instruction,
                "Reference-based evaluation", "", improved_answer,
                llm_config.evaluate_performance_client,
                llm_config.evaluate_performance_model
            )

            # Record history
            history.append({
                "iteration": iteration + 1,
                "current_answer": current_answer,
                "improved_answer": improved_answer,
                "reference_used": reference,
                "score": eval_result.total_score,
                "is_improvement": eval_result.total_score > best_score
            })

            # Update best if improved
            if eval_result.total_score > best_score:
                best_score = eval_result.total_score
                best_answer = improved_answer

            current_answer = improved_answer

        self.debug_print(f"🏁 Reference optimization completed! Best score: {best_score}")
        return best_answer, best_score, history

    async def simple_selection_loop(self, character: str, scenario: str, instruction: str,
                                  num_options: int = 5, llm_config: LLMConfig) -> Tuple[str, str, float]:
        """
        Simple selection optimization loop

        Generates multiple response options and selects the best one.

        Args:
            character: Character description
            scenario: Scenario context
            instruction: Role-playing instruction
            num_options: Number of response options to generate
            llm_config: LLM configuration

        Returns:
            Tuple of (best_response, reasoning, score)
        """
        self.debug_print(f"🎯 Starting simple selection loop...")

        base_engine = RCharengine(debug_mode=self.debug_mode)
        response_options = []

        # Generate multiple response options
        for i in range(num_options):
            self.debug_print(f"🎲 Generating option {i + 1}/{num_options}")

            result = await base_engine.execute_roleplay(
                character, scenario, instruction,
                llm_client=llm_config.execute_roleplay_client,
                model=llm_config.execute_roleplay_model
            )
            response_options.append(result.answer)

        # Select best option
        best_response, reasoning = await self.simple_selection(
            character, scenario, instruction, response_options,
            llm_config.evaluate_performance_client,
            llm_config.evaluate_performance_model
        )

        # Evaluate the selected response
        eval_result = await base_engine.evaluate_performance(
            character, scenario, instruction,
            "Simple selection evaluation", "", best_response,
            llm_config.evaluate_performance_client,
            llm_config.evaluate_performance_model
        )

        self.debug_print(f"🏁 Simple selection completed! Score: {eval_result.total_score}")
        return best_response, reasoning, eval_result.total_score

    async def adaptive_optimization_loop(self, character: str, scenario: str, instruction: str,
                                       llm_config: LLMConfig, max_iterations: int = 4) -> Tuple[str, str, float, List[Dict]]:
        """
        Adaptive optimization loop

        Combines multiple optimization strategies based on performance feedback.

        Args:
            character: Character description
            scenario: Scenario context
            instruction: Role-playing instruction
            llm_config: LLM configuration
            max_iterations: Maximum optimization iterations

        Returns:
            Tuple of (best_think, best_answer, best_score, optimization_history)
        """
        self.debug_print(f"🔄 Starting adaptive optimization loop...")

        base_engine = RCharengine(debug_mode=self.debug_mode)

        # Get initial response
        initial_result = await base_engine.execute_roleplay(
            character, scenario, instruction,
            llm_client=llm_config.execute_roleplay_client,
            model=llm_config.execute_roleplay_model
        )

        best_think = initial_result.think
        best_answer = initial_result.answer
        best_score = 0
        history = []

        for iteration in range(max_iterations):
            self.debug_print(f"📍 Adaptive iteration {iteration + 1}/{max_iterations}")

            # Choose optimization strategy based on iteration
            if iteration == 0:
                # First iteration: rewrite optimization
                new_think, new_answer = await self.rewrite_optimization(
                    character, scenario, instruction,
                    best_think, best_answer,
                    llm_config.evaluate_performance_client,
                    llm_config.evaluate_performance_model
                )
                strategy_used = "rewrite"
            elif iteration < max_iterations - 1:
                # Middle iterations: continue with standard optimization
                # (This would use the standard R-CHAR optimization loop)
                strategy_used = "standard"
                new_think, new_answer = best_think, best_answer
            else:
                # Final iteration: use best available response
                strategy_used = "final_selection"
                new_think, new_answer = best_think, best_answer

            # Evaluate current response
            eval_result = await base_engine.evaluate_performance(
                character, scenario, instruction,
                f"Adaptive iteration {iteration + 1}", new_think, new_answer,
                llm_config.evaluate_performance_client,
                llm_config.evaluate_performance_model
            )

            # Record history
            history.append({
                "iteration": iteration + 1,
                "strategy": strategy_used,
                "think": new_think,
                "answer": new_answer,
                "score": eval_result.total_score,
                "is_improvement": eval_result.total_score > best_score
            })

            # Update best if improved
            if eval_result.total_score > best_score:
                best_score = eval_result.total_score
                best_think = new_think
                best_answer = new_answer

        self.debug_print(f"🏁 Adaptive optimization completed! Best score: {best_score}")
        return best_think, best_answer, best_score, history


# Export main classes
__all__ = [
    'TrajectorySynthesis'
]