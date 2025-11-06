#!/usr/bin/env python3
"""
Evaluation Example for R-CHAR Framework

This example demonstrates how to use the evaluation tools to assess
role-playing performance on benchmark datasets.
"""

import asyncio
import json
import os
from pathlib import Path

# Add the parent directory to the path to import rchar
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rchar.evaluation.social_evaluator import SocialBenchEvaluator
from rchar.core.utils.llms import create_llm_client, create_ollama_client
from rchar.datasets.downloader import DatasetDownloader


async def social_bench_evaluation_example():
    """
    Example of evaluating model performance on SocialBench dataset
    """
    print("📊 SocialBench Evaluation Example")
    print("=" * 50)

    # Set up LLM client
    print("🤖 Setting up LLM client...")

    # Option 1: Use OpenAI API
    # openai_client = create_llm_client(
    #     base_url="https://api.openai.com/v1/",
    #     api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here")
    # )
    # model_name = "gpt-4"

    # Option 2: Use Ollama (local models)
    ollama_client = create_ollama_client("http://localhost:11434/v1/")
    model_name = "llama2"  # Change to your available model

    # Initialize evaluator
    evaluator = SocialBenchEvaluator(
        llm_client=ollama_client,
        model_name=model_name,
        debug_mode=True
    )

    # Create sample evaluation data
    print("\n📝 Creating sample evaluation data...")
    sample_data = [
        {
            "id": "sample_001",
            "role_name": "Marine Biologist",
            "role_profile": "Dr. Elena Rodriguez is a marine biologist who specializes in coral reef conservation. She is passionate about ocean protection and has a scientific, calm demeanor.",
            "conversations": "Tourist: 'I noticed some of the coral looks white and sick. What's happening to it?'",
            "options": "A) That's coral bleaching due to climate change\nB) The coral is just sleeping\nC) It's a natural cycle that happens every year\nD) I'm not sure, I'm not a marine biologist",
            "answer": "A",
            "language": "en"
        },
        {
            "id": "sample_002",
            "role_name": "Chef",
            "role_profile": "Marcus Chen is a former chef who runs a food truck. He is gruff on the outside but cares deeply about his customers. He loves fusion cuisine.",
            "conversations": "Customer: 'Your fusion dish doesn't taste authentic. Can I get a refund?'",
            "options": "A) Fine, take your money back\nB) Authenticity is boring, innovation is delicious\nC) Let me explain what makes this dish special\nD) Sorry you feel that way, here's your refund",
            "answer": "C",
            "language": "en"
        }
    ]

    print(f"📋 Created {len(sample_data)} sample evaluation items")

    # Run evaluation
    print(f"\n🚀 Starting evaluation with model: {model_name}")
    output_path = "./output/social_bench_evaluation_results.json"

    try:
        results, stats = await evaluator.evaluate_dataset(
            dataset=sample_data,
            max_concurrent=2,
            save_path=output_path,
            show_progress=True
        )

        # Display results
        print(f"\n📊 Evaluation Results:")
        print(f"   Total items: {stats['total_items']}")
        print(f"   Correct items: {stats['correct_items']}")
        print(f"   Accuracy: {stats['accuracy']:.2%}")
        print(f"   Error items: {stats['error_items']}")

        # Generate detailed report
        report_path = "./output/social_bench_evaluation_report.md"
        report = evaluator.create_evaluation_report(results, report_path)
        print(f"\n📄 Detailed report saved to: {report_path}")

    except Exception as e:
        print(f"\n❌ Evaluation error: {e}")
        print("💡 Make sure your LLM client is properly configured and accessible")


async def comparative_evaluation_example():
    """
    Example of comparing multiple models or configurations
    """
    print("\n" + "=" * 50)
    print("🔬 Comparative Evaluation Example")
    print("=" * 50)

    # Define models to compare
    models_to_compare = [
        {"name": "llama2", "client": create_ollama_client("http://localhost:11434/v1/")},
        # Add more models as needed
        # {"name": "gpt-4", "client": create_llm_client(api_key="your-key")},
    ]

    # Sample evaluation data
    sample_data = [
        {
            "id": "comp_001",
            "role_name": "Teacher",
            "role_profile": "Sarah Jenkins is an elementary school teacher who is patient and creative.",
            "conversations": "Student: 'I don't understand this math problem. It's too hard!'",
            "options": "A) Let me try explaining it differently\nB) You're right, this is very difficult\nC) Just keep trying, you'll get it eventually\nD) Maybe math isn't your strongest subject",
            "answer": "A",
            "language": "en"
        }
    ]

    comparison_results = {}

    for model_config in models_to_compare:
        model_name = model_config["name"]
        print(f"\n🤖 Evaluating model: {model_name}")

        try:
            evaluator = SocialBenchEvaluator(
                llm_client=model_config["client"],
                model_name=model_name,
                debug_mode=False
            )

            results, stats = await evaluator.evaluate_dataset(
                dataset=sample_data,
                max_concurrent=1,
                show_progress=False
            )

            comparison_results[model_name] = stats
            print(f"   ✅ {model_name}: Accuracy {stats['accuracy']:.2%}")

        except Exception as e:
            print(f"   ❌ {model_name}: Error {e}")
            comparison_results[model_name] = {"error": str(e)}

    # Display comparison summary
    print(f"\n📊 Model Comparison Summary:")
    print("-" * 30)
    for model_name, stats in comparison_results.items():
        if "error" not in stats:
            print(f"{model_name:15} | Accuracy: {stats['accuracy']:.2%} | Correct: {stats['correct_items']}/{stats['total_items']}")
        else:
            print(f"{model_name:15} | Error: {stats['error'][:30]}...")

    # Save comparison results
    comparison_path = "./output/model_comparison.json"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Comparison results saved to: {comparison_path}")


async def custom_metrics_example():
    """
    Example of using custom evaluation metrics
    """
    print("\n" + "=" * 50)
    print("📏 Custom Metrics Example")
    print("=" * 50)

    # Sample role-playing results to evaluate
    sample_results = [
        {
            "scenario": "A customer complains about food quality",
            "character": "Marcus Chen, gruff but caring chef",
            "response": "Look, I've been cooking for 20 years. If you don't like it, that's fine. But let me show you why these flavors work together.",
            "ground_truth_think": "Customer is unhappy but I need to stand by my craft while being professional",
            "score": 8
        },
        {
            "scenario": "Explaining coral bleaching to tourists",
            "character": "Dr. Elena Rodriguez, marine biologist",
            "response": "What you're seeing is coral bleaching. It's when coral expels the algae living in its tissues due to stress, primarily from warmer ocean temperatures.",
            "ground_truth_think": "Need to explain scientific concept clearly while conveying urgency",
            "score": 9
        }
    ]

    # Calculate custom metrics
    def calculate_response_diversity(responses):
        """Calculate diversity of responses using basic metrics"""
        if not responses:
            return 0

        total_length = sum(len(r.get('response', '')) for r in responses)
        avg_length = total_length / len(responses)

        # Check for unique words across responses
        all_words = set()
        for result in responses:
            words = result.get('response', '').lower().split()
            all_words.update(words)

        return {
            "avg_response_length": avg_length,
            "unique_word_count": len(all_words),
            "total_responses": len(responses)
        }

    def calculate_performance_distribution(results):
        """Calculate performance score distribution"""
        if not results:
            return {}

        scores = [r.get('score', 0) for r in results]
        return {
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "score_distribution": {
                "excellent": len([s for s in scores if s >= 9]),
                "good": len([s for s in scores if 7 <= s < 9]),
                "average": len([s for s in scores if 5 <= s < 7]),
                "poor": len([s for s in scores if s < 5])
            }
        }

    # Calculate metrics
    diversity_metrics = calculate_response_diversity(sample_results)
    performance_metrics = calculate_performance_distribution(sample_results)

    print("📈 Custom Evaluation Metrics:")
    print(f"\nResponse Diversity:")
    print(f"   Average length: {diversity_metrics['avg_response_length']:.1f} characters")
    print(f"   Unique words: {diversity_metrics['unique_word_count']}")
    print(f"   Total responses: {diversity_metrics['total_responses']}")

    print(f"\nPerformance Distribution:")
    print(f"   Average score: {performance_metrics['avg_score']:.1f}")
    print(f"   Score range: {performance_metrics['min_score']} - {performance_metrics['max_score']}")
    print(f"   Excellent (9-10): {performance_metrics['score_distribution']['excellent']}")
    print(f"   Good (7-8): {performance_metrics['score_distribution']['good']}")
    print(f"   Average (5-6): {performance_metrics['score_distribution']['average']}")
    print(f"   Poor (<5): {performance_metrics['score_distribution']['poor']}")

    # Save custom metrics
    metrics_path = "./output/custom_evaluation_metrics.json"
    metrics_data = {
        "diversity_metrics": diversity_metrics,
        "performance_metrics": performance_metrics,
        "sample_count": len(sample_results)
    }

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Custom metrics saved to: {metrics_path}")


async def main():
    """
    Main function to run all evaluation examples
    """
    print("🎯 R-CHAR Evaluation Examples")
    print("This demo showcases the evaluation capabilities of the R-CHAR framework.")
    print("\n💡 Note: Make sure you have an LLM service running for evaluation")

    # Create output directory
    Path("./output").mkdir(exist_ok=True)

    try:
        # Run SocialBench evaluation example
        await social_bench_evaluation_example()

        # Run comparative evaluation example
        await comparative_evaluation_example()

        # Run custom metrics example
        await custom_metrics_example()

        print("\n🎉 All evaluation examples completed successfully!")
        print("\n📖 Next steps:")
        print("   1. Check the output directory for evaluation results")
        print("   2. Review the generated reports and metrics")
        print("   3. Try with different models and larger datasets")
        print("   4. Create custom evaluation metrics for your use case")

    except KeyboardInterrupt:
        print("\n⚠️ Evaluation examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please check your configuration and try again")


if __name__ == "__main__":
    # Run the evaluation examples
    asyncio.run(main())