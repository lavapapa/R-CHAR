#!/usr/bin/env python3
"""
Custom Dataset Example for R-CHAR Framework

This example demonstrates how to:
- Create custom datasets
- Load and process different data formats
- Use R-CHAR with your own data
- Set up evaluation with custom data
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the parent directory to the path to import rchar
sys.path.append(str(Path(__file__).parent.parent))

from rchar.core.core_engine import RCharengine, LLMConfig
from rchar.core.utils.llms import create_ollama_client
from rchar.datasets.downloader import DatasetDownloader
from rchar.evaluation.social_evaluator import SocialBenchEvaluator


class CustomDatasetManager:
    """Manager for custom datasets and evaluation"""

    def __init__(self):
        """Initialize dataset manager"""
        self.downloader = DatasetDownloader(debug_mode=True)
        self.output_dir = "./custom_dataset_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def create_character_dataset(self, personas: List[Dict[str, Any]], output_path: str):
        """
        Create a custom character dataset

        Args:
            personas: List of persona dictionaries
            output_path: Path to save the dataset
        """
        dataset = []

        for i, persona in enumerate(personas):
            # Create multiple scenarios per character
            scenarios = self._generate_scenarios_for_persona(persona)

            for j, scenario in enumerate(scenarios):
                dataset.append({
                    "id": f"custom_{i:03d}_{j:02d}",
                    "character_name": persona.get("name", f"Character_{i}"),
                    "character": persona.get("description", ""),
                    "traits": persona.get("traits", []),
                    "scenario": scenario["scenario"],
                    "instruction": scenario["instruction"],
                    "difficulty": scenario["difficulty"],
                    "expected_behaviors": scenario.get("expected_behaviors", []),
                    "source": "custom_dataset"
                })

        # Save dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"✅ Created custom dataset with {len(dataset)} items: {output_path}")
        return dataset

    def _generate_scenarios_for_persona(self, persona: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate scenarios for a specific persona

        Args:
            persona: Persona dictionary

        Returns:
            List of scenario dictionaries
        """
        name = persona.get("name", "Character")
        traits = persona.get("traits", [])

        scenarios = []

        # Basic scenario
        scenarios.append({
            "scenario": f"{name} is in a typical daily situation when someone approaches with a simple question.",
            "instruction": "How do you respond in character?",
            "difficulty": "basic"
        })

        # Emotional scenario
        scenarios.append({
            "scenario": f"{name} faces an emotionally challenging situation that tests their core values.",
            "instruction": "How do you handle this emotional situation while staying in character?",
            "difficulty": "advanced",
            "expected_behaviors": ["emotional_intelligence", "value_consistency"]
        })

        # Professional scenario
        if any(trait in ["professional", "expert", "specialist"] for trait in traits):
            scenarios.append({
                "scenario": f"{name} is asked to demonstrate their professional expertise in a complex situation.",
                "instruction": "Show your professional knowledge and problem-solving approach.",
                "difficulty": "expert",
                "expected_behaviors": ["expertise_demonstration", "professional_behavior"]
            })

        # Social scenario
        scenarios.append({
            "scenario": f"{name} must navigate a complex social interaction involving multiple people with different perspectives.",
            "instruction": "How do you manage this social situation while maintaining your character traits?",
            "difficulty": "advanced",
            "expected_behaviors": ["social_intelligence", "character_consistency"]
        })

        return scenarios

    def create_evaluation_dataset(self, personas: List[Dict[str, Any]], output_path: str):
        """
        Create evaluation dataset with multiple choice questions

        Args:
            personas: List of persona dictionaries
            output_path: Path to save evaluation dataset
        """
        eval_dataset = []

        for i, persona in enumerate(personas):
            name = persona.get("name", f"Character_{i}")
            description = persona.get("description", "")

            # Create evaluation questions
            questions = self._generate_evaluation_questions(name, description)

            for j, question in enumerate(questions):
                eval_dataset.append({
                    "id": f"eval_{i:03d}_{j:02d}",
                    "role_name": name,
                    "role_profile": description,
                    "conversations": question["context"],
                    "options": question["options"],
                    "answer": question["correct_answer"],
                    "language": "en",
                    "difficulty": question.get("difficulty", "medium")
                })

        # Save evaluation dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_dataset, f, ensure_ascii=False, indent=2)

        print(f"✅ Created evaluation dataset with {len(eval_dataset)} items: {output_path}")
        return eval_dataset

    def _generate_evaluation_questions(self, name: str, description: str) -> List[Dict[str, Any]]:
        """
        Generate evaluation questions for a character

        Args:
            name: Character name
            description: Character description

        Returns:
            List of question dictionaries
        """
        questions = []

        # Extract key traits from description
        traits = self._extract_traits_from_description(description)

        # Basic consistency question
        if "teacher" in description.lower() or "educator" in description.lower():
            questions.append({
                "context": f"Student: 'I'm having trouble understanding this concept. Can you help?'",
                "options": [
                    "A) Let me break this down into simpler steps for you.",
                    "B) That's too bad, you should figure it out yourself.",
                    "C) This concept is actually very simple, I don't know why you're struggling.",
                    "D) Maybe this subject isn't right for you."
                ],
                "correct_answer": "A",
                "difficulty": "easy"
            })

        if "doctor" in description.lower() or "physician" in description.lower():
            questions.append({
                "context": f"Patient: 'I'm scared about the upcoming procedure. What should I expect?'",
                "options": [
                    "A) Don't worry about it, it's routine.",
                    "B) Let me explain what will happen step by step and address your concerns.",
                    "C) You should have researched this yourself before coming.",
                    "D) Many patients are scared, you'll get over it."
                ],
                "correct_answer": "B",
                "difficulty": "medium"
            })

        # General character consistency question
        questions.append({
            "context": f"Someone approaches {name} with an unusual request that challenges their typical behavior patterns.",
            "options": [
                "A) Immediately agree without consideration.",
                "B) Respond according to established character traits and values.",
                "C) Completely change personality to accommodate the request.",
                "D) Ignore the request and walk away."
            ],
            "correct_answer": "B",
            "difficulty": "medium"
        })

        return questions

    def _extract_traits_from_description(self, description: str) -> List[str]:
        """Extract character traits from description"""
        # Simple trait extraction based on keywords
        traits = []
        trait_keywords = {
            "patient": ["patient", "calm", "understanding"],
            "professional": ["professional", "expert", "specialist"],
            "caring": ["caring", "compassionate", "empathetic"],
            "analytical": ["analytical", "logical", "detail-oriented"],
            "creative": ["creative", "innovative", "artistic"]
        }

        description_lower = description.lower()
        for trait, keywords in trait_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                traits.append(trait)

        return traits


async def main():
    """Main function demonstrating custom dataset usage"""
    print("🎯 R-CHAR Custom Dataset Examples")
    print("=" * 60)

    # Initialize dataset manager
    manager = CustomDatasetManager()

    # Example 1: Create custom character dataset
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Creating Custom Character Dataset")

    custom_personas = [
        {
            "name": "Professor Evelyn Reed",
            "description": "Dr. Evelyn Reed, 65-year-old retired literature professor. Wise, patient, with a dry wit and deep love for classic novels. Spent 40 years teaching Victorian literature and can quote Shakespeare from memory.",
            "traits": ["wise", "patient", "educational", "dry_humor", "literary"]
        },
        {
            "name": "Officer Marcus Chen",
            "description": "Officer Marcus Chen, 32-year-old police officer. Dedicated, by-the-book, with a strong sense of justice. Former military police, now serving in a mid-sized city. Calm under pressure but firm when necessary.",
            "traits": ["professional", "calm_under_pressure", "by_the_book", "just"]
        },
        {
            "name": "Chef Isabella Martinez",
            "description": "Chef Isabella Martinez, 42-year-old owner of a fusion restaurant. Passionate, creative, temperamental about food quality. Trained in Paris but combines traditional techniques with local ingredients.",
            "traits": ["creative", "passionate", "professional", "temperamental"]
        },
        {
            "name": "Dr. James Wilson",
            "description": "Dr. James Wilson, 55-year-old therapist specializing in adolescent psychology. Empathetic, insightful, with a gentle approach. Known for using metaphors and storytelling in therapy sessions.",
            "traits": ["empathetic", "insightful", "gentle", "educational"]
        }
    ]

    # Create character dataset
    character_dataset_path = os.path.join(manager.output_dir, "custom_characters.json")
    dataset = manager.create_character_dataset(custom_personas, character_dataset_path)

    # Example 2: Create evaluation dataset
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Creating Custom Evaluation Dataset")

    eval_dataset_path = os.path.join(manager.output_dir, "custom_evaluation.json")
    eval_dataset = manager.create_evaluation_dataset(custom_personas, eval_dataset_path)

    # Example 3: Use R-CHAR with custom dataset
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Optimizing with Custom Dataset")

    # Initialize R-CHAR engine
    engine = RCharengine(debug_mode=True)

    # Set up LLM client
    client = create_ollama_client("http://localhost:11434/v1/")
    model_name = "qwen2.5:7b"

    llm_config = LLMConfig(
        generate_scenario_client=client,
        generate_criteria_client=client,
        execute_roleplay_client=client,
        evaluate_performance_client=client,
        generate_scenario_model=model_name,
        generate_criteria_model=model_name,
        execute_roleplay_model=model_name,
        evaluate_performance_model=model_name
    )

    # Process first character from custom dataset
    if dataset:
        first_item = dataset[0]
        print(f"\n🎭 Optimizing: {first_item['character_name']}")
        print(f"📝 Scenario: {first_item['scenario'][:100]}...")

        output_file = os.path.join(manager.output_dir, "custom_optimization_results.jsonl")

        try:
            await engine.process_persona(
                persona_data=first_item['character'],
                index=0,
                llm_config=llm_config,
                output_file=output_file
            )
            print(f"✅ Optimization completed! Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Optimization failed: {e}")

    # Example 4: Evaluate with custom evaluation dataset
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Evaluating with Custom Dataset")

    if eval_dataset:
        evaluator = SocialBenchEvaluator(
            llm_client=client,
            model_name=model_name,
            debug_mode=True
        )

        # Use first few evaluation items
        sample_eval_data = eval_dataset[:3]

        try:
            results, stats = await evaluator.evaluate_dataset(
                dataset=sample_eval_data,
                max_concurrent=2,
                save_path=os.path.join(manager.output_dir, "custom_evaluation_results.json"),
                show_progress=True
            )

            print(f"✅ Evaluation completed!")
            print(f"📊 Accuracy: {stats['accuracy']:.2%}")
            print(f"📈 Total items: {stats['total_items']}")

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")

    # Example 5: Dataset analysis
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Dataset Analysis")

    # Analyze created datasets
    manager.downloader.validate_dataset(character_dataset_path)
    char_validation = manager.downloader.validate_dataset(character_dataset_path)

    print(f"\n📊 Character Dataset Analysis:")
    print(f"   Valid: {char_validation['valid']}")
    print(f"   Total items: {char_validation['total_items']}")
    print(f"   File size: {char_validation['file_size']} bytes")

    eval_validation = manager.downloader.validate_dataset(eval_dataset_path)
    print(f"\n📊 Evaluation Dataset Analysis:")
    print(f"   Valid: {eval_validation['valid']}")
    print(f"   Total items: {eval_validation['total_items']}")
    print(f"   File size: {eval_validation['file_size']} bytes")

    print(f"\n🎉 All custom dataset examples completed!")
    print(f"📁 Check {manager.output_dir} for all generated files")


if __name__ == "__main__":
    # Run custom dataset examples
    asyncio.run(main())