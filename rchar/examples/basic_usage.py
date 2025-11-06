#!/usr/bin/env python3
"""
Basic Usage Example for R-CHAR Framework

This example demonstrates how to use the R-CHAR framework for basic
role-playing optimization with a simple persona.
"""

import asyncio
import os
from pathlib import Path

# Add the parent directory to the path to import rchar
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rchar.core.core_engine import RCharengine, LLMConfig
from rchar.core.utils.llms import create_llm_client, create_ollama_client
from rchar.datasets.downloader import DatasetDownloader


async def basic_optimization_example():
    """
    Basic example of using R-CHAR for role-playing optimization
    """
    print("🚀 R-CHAR Basic Usage Example")
    print("=" * 50)

    # Configuration
    DEBUG_MODE = True
    OUTPUT_DIR = "./output"

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Initialize the R-CHAR engine
    print("\n📋 Initializing R-CHAR engine...")
    engine = RCharengine(debug_mode=DEBUG_MODE)

    # Set up LLM clients
    print("🤖 Setting up LLM clients...")

    # Option 1: Use OpenAI API
    # openai_client = create_llm_client(
    #     base_url="https://api.openai.com/v1/",
    #     api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here")
    # )
    # model_name = "gpt-4"

    # Option 2: Use Ollama (local models)
    ollama_client = create_ollama_client("http://localhost:11434/v1/")
    model_name = "llama2"  # Change to your available model

    # Create LLM configuration
    llm_config = LLMConfig(
        generate_scenario_client=ollama_client,
        generate_criteria_client=ollama_client,
        execute_roleplay_client=ollama_client,
        evaluate_performance_client=ollama_client,
        generate_scenario_model=model_name,
        generate_criteria_model=model_name,
        execute_roleplay_model=model_name,
        evaluate_performance_model=model_name
    )

    # Example persona for optimization
    persona = """
    Dr. Elena Rodriguez, 45-year-old marine biologist specializing in coral reef conservation.
    She is passionate about ocean protection, scientifically rigorous but also emotionally
    connected to marine life. She becomes very enthusiastic when discussing coral ecosystems
    and marine conservation efforts. She has a calm, patient demeanor but can be very
    persuasive when advocating for environmental protection.
    """

    print(f"\n👤 Processing persona: Dr. Elena Rodriguez...")
    print(f"📝 Model: {model_name}")

    # Process the persona through R-CHAR optimization
    output_file = os.path.join(OUTPUT_DIR, "basic_optimization_result.jsonl")

    try:
        await engine.process_persona(
            persona_data=persona,
            index=1,
            llm_config=llm_config,
            output_file=output_file
        )

        print(f"\n✅ Optimization completed!")
        print(f"📁 Results saved to: {output_file}")

        # Read and display results summary
        if os.path.exists(output_file):
            import json
            with open(output_file, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f if line.strip()]

            print(f"\n📊 Results Summary:")
            print(f"   Total scenarios processed: {len(results)}")

            for i, result in enumerate(results):
                print(f"\n   Scenario {i+1}:")
                print(f"   - Level: {result.get('level', 'unknown')}")
                print(f"   - Final Score: {result.get('final_score', 0)}")
                print(f"   - Iterations: {result.get('iterations', 0)}")
                print(f"   - Scenario: {result.get('scenario', 'N/A')[:100]}...")

    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        print("💡 Make sure your LLM client is properly configured and accessible")


async def dataset_example():
    """
    Example of using dataset downloader and loader
    """
    print("\n" + "=" * 50)
    print("📚 Dataset Example")
    print("=" * 50)

    # Initialize dataset downloader
    downloader = DatasetDownloader(debug_mode=True)

    # Create a sample dataset for testing
    print("\n📝 Creating sample dataset...")
    sample_path = downloader.create_sample_dataset(
        output_path="./datasets/sample_personas.json",
        num_samples=5
    )

    print(f"✅ Sample dataset created: {sample_path}")

    # Validate the dataset
    print("\n🔍 Validating dataset...")
    validation_result = downloader.validate_dataset(sample_path)

    if validation_result["valid"]:
        print("✅ Dataset is valid!")
        print(f"   Total items: {validation_result['total_items']}")
        print(f"   File format: {validation_result['file_format']}")
        print(f"   File size: {validation_result['file_size']} bytes")
    else:
        print(f"❌ Dataset validation failed: {validation_result['error']}")


async def trajectory_synthesis_example():
    """
    Example of using alternative trajectory synthesis methods
    """
    print("\n" + "=" * 50)
    print("🔄 Trajectory Synthesis Example")
    print("=" * 50)

    from rchar.core.trajectory_synthesis import TrajectorySynthesis

    # Initialize trajectory synthesis
    trajectory = TrajectorySynthesis(debug_mode=True)

    # Example persona and scenario
    persona = "Marcus Chen, 42-year-old former chef turned food truck owner. Gruff exterior but deeply cares about customers. Loves fusion cuisine and has sarcastic wit."
    scenario = "A customer complains that their fusion dish is 'not authentic' and demands a refund."
    instruction = "How do you respond while maintaining your restaurant's reputation and customer satisfaction?"

    # Set up LLM client
    ollama_client = create_ollama_client("http://localhost:11434/v1/")

    print("\n🎭 Testing rewrite optimization...")

    try:
        # Create initial response
        engine = RCharengine(debug_mode=False)
        initial_result = await engine.execute_roleplay(
            persona, scenario, instruction,
            llm_client=ollama_client,
            model="llama2"
        )

        print(f"📝 Initial response: {initial_result.answer[:200]}...")

        # Apply rewrite optimization
        rewritten_think, rewritten_answer = await trajectory.rewrite_optimization(
            persona, scenario, instruction,
            initial_result.think, initial_result.answer,
            ollama_client, "llama2"
        )

        print(f"\n✨ Rewritten response: {rewritten_answer[:200]}...")

    except Exception as e:
        print(f"❌ Trajectory synthesis error: {e}")


async def main():
    """
    Main function to run all examples
    """
    print("🎯 R-CHAR Framework Examples")
    print("This demo showcases the main capabilities of the R-CHAR framework.")
    print("\n💡 Note: Make sure you have an LLM service running (OpenAI API or Ollama)")

    try:
        # Run basic optimization example
        await basic_optimization_example()

        # Run dataset example
        await dataset_example()

        # Run trajectory synthesis example
        await trajectory_synthesis_example()

        print("\n🎉 All examples completed successfully!")
        print("\n📖 Next steps:")
        print("   1. Check the output directory for results")
        print("   2. Review the generated optimization data")
        print("   3. Try with your own personas and scenarios")
        print("   4. Explore advanced examples in the examples/ directory")

    except KeyboardInterrupt:
        print("\n⚠️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please check your configuration and try again")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())