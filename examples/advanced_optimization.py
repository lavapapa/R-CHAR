#!/usr/bin/env python3
"""
Advanced Optimization Example for R-CHAR Framework

This example demonstrates advanced usage including:
- Custom configuration
- Batch processing
- Multiple optimization strategies
- Ablation studies
- Performance comparison
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import yaml

# Add the parent directory to the path to import rchar
sys.path.append(str(Path(__file__).parent.parent))

from rchar.core.core_engine import RCharengine, LLMConfig
from rchar.core.trajectory_synthesis import TrajectorySynthesis
from rchar.core.utils.llms import create_ollama_client, create_llm_client
from rchar.datasets.downloader import DatasetDownloader
from rchar.evaluation.social_evaluator import SocialBenchEvaluator


class AdvancedOptimization:
    """Advanced optimization workflow manager"""

    def __init__(self, config_path: str = None):
        """
        Initialize advanced optimization

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()

        # Initialize components
        self.engine = RCharengine(
            config=self.config.get('optimization', {}),
            debug_mode=self.config.get('optimization', {}).get('debug_mode', False)
        )

        self.trajectory = TrajectorySynthesis(
            config=self.config.get('optimization', {}),
            debug_mode=self.config.get('optimization', {}).get('debug_mode', False)
        )

        # Set up LLM client
        self.client = self._setup_llm_client()
        self.model_name = self.config.get('llm', {}).get('model', 'qwen2.5:7b')

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'optimization': {
                'max_iterations': 3,
                'max_roleplay_iterations': 2,
                'debug_mode': True
            },
            'llm': {
                'type': 'ollama',
                'base_url': 'http://localhost:11434/v1/',
                'model': 'qwen2.5:7b'
            },
            'output': {
                'base_dir': './advanced_output'
            }
        }

    def _setup_llm_client(self):
        """Set up LLM client based on configuration"""
        llm_config = self.config.get('llm', {})
        llm_type = llm_config.get('type', 'ollama')

        if llm_type == 'ollama':
            return create_ollama_client(llm_config.get('base_url', 'http://localhost:11434/v1/'))
        elif llm_type == 'openai':
            return create_llm_client(
                base_url=llm_config.get('base_url', 'https://api.openai.com/v1/'),
                api_key=os.getenv('OPENAI_API_KEY', 'your-api-key')
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    def create_llm_config(self) -> LLMConfig:
        """Create LLM configuration"""
        return LLMConfig(
            generate_scenario_client=self.client,
            generate_criteria_client=self.client,
            execute_roleplay_client=self.client,
            evaluate_performance_client=self.client,
            generate_scenario_model=self.model_name,
            generate_criteria_model=self.model_name,
            execute_roleplay_model=self.model_name,
            evaluate_performance_model=self.model_name
        )

    async def comparative_optimization(self, persona: str, scenario: str, instruction: str):
        """
        Compare different optimization strategies

        Args:
            persona: Character description
            scenario: Scenario context
            instruction: Role-playing instruction
        """
        print("🔬 Comparative Optimization Study")
        print("=" * 50)

        results = {}

        # Strategy 1: Standard R-CHAR
        print("\n1️⃣ Standard R-CHAR Optimization...")
        try:
            think_rchar, answer_rchar, score_rchar, _, procedure_rchar, _ = await self.engine.optimization_loop(
                persona, scenario, instruction, "Custom evaluation criteria", self.create_llm_config()
            )
            results['rchar'] = {
                'think': think_rchar,
                'answer': answer_rchar,
                'score': score_rchar,
                'iterations': len(procedure_rchar)
            }
            print(f"   ✅ R-CHAR Score: {score_rchar}")
        except Exception as e:
            print(f"   ❌ R-CHAR Error: {e}")
            results['rchar'] = None

        # Strategy 2: Reference Optimization
        print("\n2️⃣ Reference-based Optimization...")
        try:
            # Create a baseline response
            baseline_result = await self.engine.execute_roleplay(
                persona, scenario, instruction, llm_client=self.client, model=self.model_name
            )

            # Create a high-quality reference (simplified for example)
            reference = f"As {persona.split(',')[0] if ',' in persona else persona.split('.')[0]}, I would respond with thoughtful consideration of the situation, drawing from my experience and character traits."

            improved_answer = await self.trajectory.reference_optimization(
                persona, scenario, instruction,
                baseline_result.answer, reference,
                self.client, self.model_name
            )
            results['reference'] = {
                'answer': improved_answer,
                'baseline': baseline_result.answer,
                'reference': reference
            }
            print(f"   ✅ Reference optimization completed")
        except Exception as e:
            print(f"   ❌ Reference Error: {e}")
            results['reference'] = None

        # Strategy 3: Simple Selection
        print("\n3️⃣ Simple Selection Strategy...")
        try:
            # Generate multiple options
            options = []
            for i in range(3):
                result = await self.engine.execute_roleplay(
                    persona, scenario, instruction, llm_client=self.client, model=self.model_name
                )
                options.append(result.answer)

            best_answer, reasoning = await self.trajectory.simple_selection(
                persona, scenario, instruction, options,
                self.client, self.model_name
            )
            results['selection'] = {
                'best_answer': best_answer,
                'reasoning': reasoning,
                'all_options': options
            }
            print(f"   ✅ Simple selection completed")
        except Exception as e:
            print(f"   ❌ Selection Error: {e}")
            results['selection'] = None

        # Save comparison results
        output_dir = self.config.get('output', {}).get('base_dir', './advanced_output')
        os.makedirs(output_dir, exist_ok=True)

        comparison_file = os.path.join(output_dir, 'optimization_comparison.json')
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n📁 Comparison results saved to: {comparison_file}")

        return results

    async def batch_character_optimization(self, personas: List[Dict[str, Any]]):
        """
        Optimize multiple characters in batch

        Args:
            personas: List of persona dictionaries
        """
        print(f"🚀 Batch Optimization: {len(personas)} characters")
        print("=" * 50)

        llm_config = self.create_llm_config()
        output_dir = self.config.get('output', {}).get('base_dir', './advanced_output')
        output_file = os.path.join(output_dir, 'batch_optimization_results.jsonl')

        results = []

        for i, persona_data in enumerate(personas):
            print(f"\n👤 Processing character {i+1}/{len(personas)}: {persona_data.get('name', 'Unknown')}")

            try:
                await self.engine.process_persona(
                    persona_data=persona_data.get('description', ''),
                    index=i,
                    llm_config=llm_config,
                    output_file=output_file
                )
                print(f"   ✅ Completed character {i+1}")
            except Exception as e:
                print(f"   ❌ Error processing character {i+1}: {e}")

        print(f"\n📁 Batch results saved to: {output_file}")

    async def performance_profiling(self):
        """
        Profile performance across different scenarios
        """
        print("⚡ Performance Profiling")
        print("=" * 50)

        # Test persona
        test_persona = """
        Dr. Alex Chen, 38-year-old AI researcher specializing in natural language processing.
        Analytical, detail-oriented, passionate about ethical AI development.
        """

        # Test scenarios of varying complexity
        scenarios = [
            {
                'name': 'Simple Question',
                'scenario': 'A student asks what NLP is.',
                'instruction': 'Explain NLP in simple terms.',
                'complexity': 'basic'
            },
            {
                'name': 'Ethical Dilemma',
                'scenario': 'A company wants to deploy an AI system that may have privacy implications.',
                'instruction': 'Provide ethical guidance on the deployment.',
                'complexity': 'advanced'
            },
            {
                'name': 'Technical Challenge',
                'scenario': 'A complex NLP model is producing biased outputs.',
                'instruction': 'Diagnose the issue and propose solutions.',
                'complexity': 'expert'
            }
        ]

        profiling_results = []

        for scenario_data in scenarios:
            print(f"\n🧪 Testing {scenario_data['name']} ({scenario_data['complexity']})...")

            import time
            start_time = time.time()

            try:
                think, answer, score, iterations, procedure, _ = await self.engine.optimization_loop(
                    test_persona,
                    scenario_data['scenario'],
                    scenario_data['instruction'],
                    f"Professional and ethical response criteria",
                    self.create_llm_config()
                )

                end_time = time.time()
                duration = end_time - start_time

                result = {
                    'scenario': scenario_data['name'],
                    'complexity': scenario_data['complexity'],
                    'score': score,
                    'iterations': iterations,
                    'duration_seconds': duration,
                    'response_length': len(answer),
                    'thinking_length': len(think)
                }

                profiling_results.append(result)
                print(f"   ✅ Score: {score}, Time: {duration:.2f}s, Iterations: {iterations}")

            except Exception as e:
                print(f"   ❌ Error: {e}")

        # Save profiling results
        output_dir = self.config.get('output', {}).get('base_dir', './advanced_output')
        profiling_file = os.path.join(output_dir, 'performance_profile.json')

        with open(profiling_file, 'w', encoding='utf-8') as f:
            json.dump(profiling_results, f, ensure_ascii=False, indent=2)

        print(f"\n📊 Profiling results saved to: {profiling_file}")

        # Print summary statistics
        if profiling_results:
            avg_score = sum(r['score'] for r in profiling_results) / len(profiling_results)
            avg_time = sum(r['duration_seconds'] for r in profiling_results) / len(profiling_results)
            print(f"\n📈 Summary Statistics:")
            print(f"   Average Score: {avg_score:.1f}")
            print(f"   Average Time: {avg_time:.2f}s")
            print(f"   Total Scenarios: {len(profiling_results)}")


async def main():
    """Main function demonstrating advanced features"""
    print("🎯 R-CHAR Advanced Optimization Examples")
    print("=" * 60)

    # Initialize advanced optimization
    advanced = AdvancedOptimization()

    # Create output directory
    output_dir = advanced.config.get('output', {}).get('base_dir', './advanced_output')
    os.makedirs(output_dir, exist_ok=True)

    # Example 1: Comparative optimization
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Comparative Optimization Strategies")
    await advanced.comparative_optimization(
        persona="Dr. Sarah Mitchell, marine biologist, passionate about conservation",
        scenario="A tourist asks if coral bleaching is really that serious",
        instruction="Respond as a concerned but educational marine biologist"
    )

    # Example 2: Batch processing
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Character Optimization")

    # Create sample personas for batch processing
    sample_personas = [
        {
            'name': 'Teacher',
            'description': 'Ms. Jennifer Brown, high school literature teacher, patient and inspiring'
        },
        {
            'name': 'Chef',
            'description': 'Chef Marco Rossi, Italian chef, passionate about traditional cooking'
        },
        {
            'name': 'Doctor',
            'description': 'Dr. Michael Lee, emergency room physician, calm under pressure'
        }
    ]

    await advanced.batch_character_optimization(sample_personas)

    # Example 3: Performance profiling
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Performance Profiling")
    await advanced.performance_profiling()

    print("\n🎉 All advanced examples completed!")
    print(f"📁 Check {output_dir} for detailed results")


if __name__ == "__main__":
    # Run advanced examples
    asyncio.run(main())