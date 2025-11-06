"""
R-CHAR Core Engine

This module implements the core optimization algorithm for role-playing enhancement
through thinking trajectory guidance and adaptive evaluation.
"""

from typing import Tuple, NamedTuple, List, Dict, Any, Optional
import asyncio
import json
import os
from openai import AsyncClient
from tenacity import retry, stop_after_attempt

from .utils.llms import ask, ask_json, ask_messages
from .utils.response_decoders import extract_tags_dict, extract_key_values
from .utils.async_worker import worker, create_worker_pool
from .utils.jsonl import read_jsonl
from .prompts import (
    ROLEPLAY_SYSTEM_PROMPT, ROLEPLAY_USER_PROMPT,
    PERSONA_TO_SCENARIO_PROMPT, CRITERIA_GENERATE_PROMPT,
    EVALUATE_ANALYSIS_PROMPT, GENERATE_INTROSPECTIVE_PROMPT
)


# Default configuration
DEFAULT_CONFIG = {
    "max_iterations": 4,
    "max_roleplay_iterations": 3,
    "max_scenarios_per_persona": 5,
    "debug_mode": False
}


class ScenarioResult(NamedTuple):
    """Result of scenario generation"""
    character: str
    scenario: str
    instruction: str
    level: str  # basic, advanced, difficult


class RoleplayResult(NamedTuple):
    """Result of role-playing execution"""
    think: str
    answer: str


class EvaluateResult(NamedTuple):
    """Result of performance evaluation"""
    overall_flaws: str
    criteria_evaluation: Dict[str, str]
    total_score: int


class LLMConfig(NamedTuple):
    """Configuration for LLM clients"""
    generate_scenario_client: AsyncClient
    generate_criteria_client: AsyncClient
    execute_roleplay_client: AsyncClient
    evaluate_performance_client: AsyncClient
    generate_scenario_model: str
    generate_criteria_model: str
    execute_roleplay_model: str
    evaluate_performance_model: str


class RCharengine:
    """Main engine for R-CHAR optimization framework"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, debug_mode: bool = False):
        """
        Initialize R-CHAR engine

        Args:
            config: Configuration dictionary
            debug_mode: Enable debug logging
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.debug_mode = debug_mode

    def debug_print(self, *args, **kwargs):
        """Print debug information if debug mode is enabled"""
        if self.debug_mode:
            print("🔍 DEBUG:", *args, **kwargs)

    @retry(stop=stop_after_attempt(3))
    async def generate_scenario(self, persona: str, llm_client: AsyncClient, model: str) -> List[ScenarioResult]:
        """
        Generate scenarios and instructions based on persona description

        Args:
            persona: Character description
            llm_client: LLM client instance
            model: Model name

        Returns:
            List of generated scenarios
        """
        self.debug_print(f"🎭 Generating scenarios for persona: {persona[:50]}...")

        prompt = PERSONA_TO_SCENARIO_PROMPT.format(persona=persona)
        response = await ask_json(prompt, model, llm_client)

        scenarios = response if isinstance(response, list) else response.get('scenarios', [])

        result = []
        for scenario in scenarios[:self.config['max_scenarios_per_persona']]:
            if 'scenario' in scenario and 'instruction' in scenario:
                scenario_result = ScenarioResult(
                    character=persona,
                    scenario=scenario['scenario'],
                    instruction=scenario['instruction'],
                    level=scenario.get('level', 'unknown'),
                )
                result.append(scenario_result)

        self.debug_print(f"✅ Generated {len(result)} scenarios")
        return result

    @retry(stop=stop_after_attempt(3))
    async def generate_criteria(self, character: str, scenario: str, instruction: str,
                               llm_client: AsyncClient, model: str) -> str:
        """
        Generate evaluation criteria for role-playing scenario

        Args:
            character: Character description
            scenario: Scenario context
            instruction: Role-playing instruction
            llm_client: LLM client instance
            model: Model name

        Returns:
            Generated evaluation criteria
        """
        self.debug_print(f"📋 Generating evaluation criteria...")

        prompt = CRITERIA_GENERATE_PROMPT.format(
            character=character,
            scenario=scenario,
            instruction=instruction
        )
        response = await ask(prompt, model, llm_client)

        self.debug_print(f"✅ Criteria generated: {response[:50]}...")
        return response

    @retry(stop=stop_after_attempt(3))
    async def execute_roleplay(self, character: str, scenario: str, instruction: str,
                              previous_think: str = "", llm_client: Optional[AsyncClient] = None,
                              model: str = "gpt-4") -> RoleplayResult:
        """
        Execute role-playing with thinking trajectory

        Args:
            character: Character description
            scenario: Scenario context
            instruction: Role-playing instruction
            previous_think: Previous thinking process for continuation
            llm_client: LLM client instance
            model: Model name

        Returns:
            Role-playing result with thinking and answer
        """
        self.debug_print(f"🎬 Executing role-play...")

        user_prompt = ROLEPLAY_USER_PROMPT.format(
            character=character,
            scenario=scenario,
            instruction=instruction
        )

        messages = [
            {"role": "system", "content": ROLEPLAY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        if previous_think:
            continue_prompt = f"</think>{previous_think}\n(CONTINUE YOUR THINKING...)\n"
            messages.append({
                "role": "assistant",
                "content": continue_prompt
            })
            self.debug_print(f"💭 Using previous thinking: {previous_think[:50]}...")

        response = await ask_messages(messages, model, llm_client)

        if previous_think and "think" not in response.lower():
            # Ensure think tag is present
            response = f"</think>{previous_think}</think>" + response

        result = extract_tags_dict(response)

        self.debug_print(f"✨ Role-play completed")
        self.debug_print(f"💭 Think: {result['think'][:50]}...")
        self.debug_print(f"💬 Answer: {result['answer'][:50]}...")

        return RoleplayResult(think=result['think'], answer=result['answer'])

    @retry(stop=stop_after_attempt(3))
    async def evaluate_performance(self, character: str, scenario: str, instruction: str,
                                  criteria: str, think: str, answer: str,
                                  llm_client: AsyncClient, model: str) -> EvaluateResult:
        """
        Evaluate role-playing performance

        Args:
            character: Character description
            scenario: Scenario context
            instruction: Role-playing instruction
            criteria: Evaluation criteria
            think: Thinking process
            answer: Generated answer
            llm_client: LLM client instance
            model: Model name

        Returns:
            Evaluation result
        """
        self.debug_print(f"📊 Evaluating performance...")

        prompt = EVALUATE_ANALYSIS_PROMPT.format(
            character=character,
            scenario=scenario,
            instruction=instruction,
            criteria=criteria,
            think=think,
            answer=answer
        )

        response = await ask(prompt, model, llm_client)
        eval_result = extract_tags_dict(response)

        try:
            scores = extract_key_values(eval_result['criteria_evaluation'])
            self.debug_print(f"💯 Score details: {scores}")
        except Exception as e:
            self.debug_print(f"❌ Score parsing failed: {e}")
            raise ValueError(f"Failed to parse evaluation scores: {e}")

        total_score = int(sum(map(float, scores.values())))
        self.debug_print(f"🎯 Total score: {total_score}")

        return EvaluateResult(
            overall_flaws=eval_result.get('overall_flaws', ""),
            criteria_evaluation=scores,
            total_score=total_score
        )

    @retry(stop=stop_after_attempt(3))
    async def generate_introspection(self, character: str, scenario: str, instruction: str,
                                    criteria: str, think: str, overall_flaws: str,
                                    criteria_evaluation: Dict[str, str],
                                    llm_client: AsyncClient, model: str) -> str:
        """
        Generate introspective guidance for improvement

        Args:
            character: Character description
            scenario: Scenario context
            instruction: Role-playing instruction
            criteria: Evaluation criteria
            think: Current thinking process
            overall_flaws: Overall performance flaws
            criteria_evaluation: Detailed criteria evaluation
            llm_client: LLM client instance
            model: Model name

        Returns:
            Generated introspective guidance
        """
        self.debug_print(f"🤔 Generating introspective guidance...")

        criteria_text = "\n".join(f"{k}: {v}" for k, v in criteria_evaluation.items())

        prompt = GENERATE_INTROSPECTIVE_PROMPT.format(
            character=character,
            scenario=scenario,
            instruction=instruction,
            criteria=criteria,
            think=think,
            overall_flaws=overall_flaws,
            criteria_evaluation=criteria_text
        )

        response = await ask(prompt, model, llm_client)
        result = extract_tags_dict(response)
        guidance = result.get('guidance', '')

        self.debug_print(f"💡 Introspection generated: {guidance[:50]}...")
        return guidance

    async def optimization_loop(self, character: str, scenario: str, instruction: str,
                               criteria: str, llm_config: LLMConfig) -> Tuple[str, str, float, int, List[Dict], List[Dict]]:
        """
        Main optimization loop for role-playing enhancement

        Args:
            character: Character description
            scenario: Scenario context
            instruction: Role-playing instruction
            criteria: Evaluation criteria
            llm_config: LLM configuration

        Returns:
            Tuple of (best_think, best_answer, best_score, iterations, procedure, introspections)
        """
        self.debug_print(f"🔄 Starting optimization loop...")

        best_think = ""
        best_answer = ""
        best_score = 0
        best_eval_res = None
        current_think = ""
        iteration = 0
        procedure = []
        introspections = []

        max_iterations = self.config['max_iterations']
        max_roleplay_iterations = self.config['max_roleplay_iterations']

        while iteration < max_iterations:
            iteration += 1
            self.debug_print(f"📍 Iteration {iteration}/{max_iterations}")

            inner_best_think = ""
            inner_best_answer = ""
            inner_best_score = 0
            inner_best_eval = None
            inner_best_iteration = 0

            # Multiple role-playing attempts per iteration
            for inner_iter in range(max_roleplay_iterations):
                self.debug_print(f"  🔄 Inner iteration {inner_iter + 1}/{max_roleplay_iterations}")

                try:
                    roleplay_result = await self.execute_roleplay(
                        character, scenario, instruction, current_think,
                        llm_config.execute_roleplay_client,
                        llm_config.execute_roleplay_model
                    )

                    eval_res = await self.evaluate_performance(
                        character, scenario, instruction, criteria,
                        roleplay_result.think, roleplay_result.answer,
                        llm_config.evaluate_performance_client,
                        llm_config.evaluate_performance_model
                    )
                except Exception as e:
                    self.debug_print(f"❌ Role-play failed: {e}")
                    continue

                procedure.append({
                    "iteration": iteration,
                    "roleplay_iteration": inner_iter + 1,
                    "think": roleplay_result.think,
                    "answer": roleplay_result.answer,
                    "score": eval_res.total_score,
                    "evaluation": eval_res._asdict()
                })

                if eval_res.total_score > inner_best_score:
                    self.debug_print(f"  ⭐ Found better result! Score: {eval_res.total_score}")
                    inner_best_score = eval_res.total_score
                    inner_best_think = roleplay_result.think
                    inner_best_answer = roleplay_result.answer
                    inner_best_eval = eval_res
                    inner_best_iteration = inner_iter + 1

            # Check if improvement was made
            is_improved = inner_best_score > best_score
            if is_improved:
                self.debug_print(f"  ✅ Improved! New score: {inner_best_score}, Old score: {best_score}")
                best_score = inner_best_score
                best_think = inner_best_think
                best_answer = inner_best_answer
                best_eval_res = inner_best_eval

            # Generate introspection for next iteration
            introspection = await self.generate_introspection(
                character, scenario, instruction, criteria,
                best_think, best_eval_res.overall_flaws,
                best_eval_res.criteria_evaluation,
                llm_config.evaluate_performance_client,
                llm_config.evaluate_performance_model
            )

            introspections.append({
                "iteration": iteration,
                "inner_iteration": inner_best_iteration,
                "introspection": introspection,
                "is_improved": is_improved
            })

            current_think = best_think + "\n" + introspection

        self.debug_print(f"🏁 Optimization completed! Final score: {best_score}")
        return best_think, best_answer, best_score, iteration, procedure, introspections

    async def process_persona(self, persona_data: Any, index: int, llm_config: LLMConfig,
                             output_file: str) -> None:
        """
        Process a single persona through the complete optimization pipeline

        Args:
            persona_data: Persona data (string or dict)
            index: Data index for tracking
            llm_config: LLM configuration
            output_file: Output file path for results
        """
        self.debug_print(f"🎯 Processing persona at index {index}...")

        # Generate or use existing scenarios
        if isinstance(persona_data, str):
            scenarios = await self.generate_scenario(
                persona_data,
                llm_config.generate_scenario_client,
                llm_config.generate_scenario_model
            )
        elif isinstance(persona_data, dict):
            scenarios = [ScenarioResult(
                character=persona_data.get('character', ''),
                scenario=persona_data.get('scenario', ''),
                instruction=persona_data.get('instruction', ''),
                level='unknown'
            )]
        else:
            raise ValueError(f"Unsupported data type: {type(persona_data)}")

        # Process each scenario
        for sub_index, scenario_info in enumerate(scenarios):
            self.debug_print(f"Processing scenario {sub_index + 1}/{len(scenarios)}")

            # Generate evaluation criteria
            criteria = await self.generate_criteria(
                scenario_info.character,
                scenario_info.scenario,
                scenario_info.instruction,
                llm_config.generate_criteria_client,
                llm_config.generate_criteria_model
            )

            # Run optimization loop
            best_think, best_answer, final_score, iterations, procedure, introspections = await self.optimization_loop(
                scenario_info.character,
                scenario_info.scenario,
                scenario_info.instruction,
                criteria,
                llm_config
            )

            # Save results
            result = {
                "index": index,
                "sub_index": sub_index,
                "level": scenario_info.level,
                "criteria": criteria,
                "character": scenario_info.character,
                "scenario": scenario_info.scenario,
                "instruction": scenario_info.instruction,
                "best_think": best_think,
                "best_answer": best_answer,
                "final_score": final_score,
                "iterations": iterations,
                "procedure": procedure,
                "introspections": introspections
            }

            self.debug_print(f"💾 Saving result to {output_file}")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")

        self.debug_print(f"✅ Persona {index} processing completed!")


# Export main classes and functions
__all__ = [
    'RCharengine',
    'ScenarioResult',
    'RoleplayResult',
    'EvaluateResult',
    'LLMConfig',
    'DEFAULT_CONFIG'
]