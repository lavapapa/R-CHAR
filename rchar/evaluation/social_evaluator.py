"""
SocialBench Evaluator

This module provides integration with SocialBench benchmark for evaluating
social intelligence and role-playing capabilities in LLMs.
"""

import asyncio
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from openai import AsyncClient
from tqdm.asyncio import tqdm

from ..core.utils.llms import ask_messages, create_llm_client
from ..core.utils.async_worker import create_worker_pool, task_fn


class SocialBenchEvaluator:
    """
    Evaluator for SocialBench benchmark tests

    Provides comprehensive evaluation of social intelligence, role-playing
    consistency, and interaction quality across various scenarios.
    """

    def __init__(self, llm_client: AsyncClient, model_name: str, debug_mode: bool = False):
        """
        Initialize SocialBench evaluator

        Args:
            llm_client: LLM client for evaluation
            model_name: Model name to evaluate
            debug_mode: Enable debug logging
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.debug_mode = debug_mode

    def debug_print(self, *args, **kwargs):
        """Print debug information if debug mode is enabled"""
        if self.debug_mode:
            print("🔍 DEBUG:", *args, **kwargs)

    def _format_english_prompt(self, role_profile: str, conversations: str,
                              role_name: str, options: str) -> str:
        """Format English evaluation prompt"""
        return f"""==Profile==
{role_profile}

==Conversations==
{conversations}

You are playing the role of {role_name}, you need to embody the knowledge and style of {role_name}.
Based on the provided role Profile and Conversations, please choose the best option (A, B, C, or D):
{options}

Your selection:
"""

    def _format_chinese_prompt(self, role_profile: str, conversations: str,
                              role_name: str, options: str) -> str:
        """Format Chinese evaluation prompt"""
        return f"""==角色描述==
{role_profile}

==对话历史==
{conversations}

你要扮演{role_name}角色，你在聊天中要具备该角色对应的知识背景，语气风格等特征。
请根据所给的{role_name}角色描述和对话历史，从下面四个选项（A. B. C.和D.）中选择符合{role_name}的选项：
{options}

你的选择：
"""

    def extract_choice(self, response: str) -> Optional[str]:
        """
        Extract choice letter from LLM response

        Args:
            response: LLM response text

        Returns:
            Extracted choice (A, B, C, D) or None
        """
        # Look for patterns like "A.", "选择：A", "Answer: A" etc.
        patterns = [
            r'(?:选择[:：]?\s*|Answer[:：]?\s*|选择是[:：]?\s*)?([ABCD])[.。]?',
            r'([ABCD])[.。]?\s*(?:选项|option)',
            r'选择.*?([ABCD])',
            r'答案.*?([ABCD])'
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None

    @task_fn
    async def evaluate_single_item(self, data_item: Dict[str, Any],
                                  progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Evaluate a single SocialBench item

        Args:
            data_item: Single test case data
            progress_callback: Optional progress callback

        Returns:
            Evaluation result dictionary
        """
        try:
            # Extract data fields
            role_profile = data_item.get('role_profile', '')
            conversations = data_item.get('conversations', '')
            role_name = data_item.get('role_name', '')
            options = data_item.get('options', '')
            language = data_item.get('language', 'en')
            ground_truth = data_item.get('answer', '')
            item_id = data_item.get('id', '')

            self.debug_print(f"🎯 Evaluating item: {item_id}")

            # Format prompt based on language
            if language == 'zh':
                prompt = self._format_chinese_prompt(role_profile, conversations, role_name, options)
            else:
                prompt = self._format_english_prompt(role_profile, conversations, role_name, options)

            # Create messages for LLM
            messages = [{"role": "user", "content": prompt}]

            # Get model response
            response = await ask_messages(messages, self.model_name, self.llm_client)

            # Extract choice
            predicted_choice = self.extract_choice(response)

            # Determine correctness
            is_correct = predicted_choice == ground_truth.upper() if ground_truth else False

            result = {
                'id': item_id,
                'question': data_item,
                'model_response': response,
                'predicted_choice': predicted_choice,
                'ground_truth': ground_truth,
                'is_correct': is_correct,
                'language': language,
                'model_name': self.model_name
            }

            if progress_callback:
                progress_callback()

            self.debug_print(f"✅ Item {item_id}: {predicted_choice} (correct: {is_correct})")

            return result

        except Exception as e:
            self.debug_print(f"❌ Error evaluating item: {e}")
            return {
                'id': data_item.get('id', ''),
                'error': str(e),
                'is_correct': False,
                'model_name': self.model_name
            }

    async def evaluate_dataset(self, dataset: List[Dict[str, Any]],
                              max_concurrent: int = 5,
                              save_path: Optional[str] = None,
                              show_progress: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Evaluate entire SocialBench dataset

        Args:
            dataset: List of test items
            max_concurrent: Maximum concurrent evaluations
            save_path: Optional path to save results
            show_progress: Show progress bar

        Returns:
            Tuple of (results, statistics)
        """
        self.debug_print(f"🚀 Starting evaluation of {len(dataset)} items")

        # Create worker pool
        pool = create_worker_pool()
        pool.add_worker(max_concurrent, evaluator=self)

        # Progress tracking
        completed = 0
        def progress_callback():
            nonlocal completed
            completed += 1

        # Assign all tasks
        for item in dataset:
            pool.assign(self.evaluate_single_item, item, progress_callback)

        # Create progress bar
        if show_progress:
            pbar = tqdm(total=len(dataset), desc=f"📊 Evaluating {self.model_name}")
        else:
            pbar = None

        # Run evaluation
        async def run_with_progress():
            await pool.run(verbose=False)
            if pbar:
                pbar.close()

        await run_with_progress()

        # Collect results
        results = []
        # Note: In a real implementation, we'd need to collect results from workers
        # For now, we'll simulate this
        for item in dataset:
            result = await self.evaluate_single_item(item)
            results.append(result)

        # Calculate statistics
        stats = self.calculate_statistics(results)

        # Save results if path provided
        if save_path:
            self.save_results(results, save_path)
            self.debug_print(f"💾 Results saved to {save_path}")

        self.debug_print(f"🎉 Evaluation completed! Accuracy: {stats['accuracy']:.2%}")

        return results, stats

    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate evaluation statistics

        Args:
            results: List of evaluation results

        Returns:
            Statistics dictionary
        """
        total_items = len(results)
        correct_items = sum(1 for r in results if r.get('is_correct', False))
        error_items = sum(1 for r in results if 'error' in r)

        # Language-specific stats
        english_results = [r for r in results if r.get('language') == 'en']
        chinese_results = [r for r in results if r.get('language') == 'zh']

        english_correct = sum(1 for r in english_results if r.get('is_correct', False))
        chinese_correct = sum(1 for r in chinese_results if r.get('is_correct', False))

        stats = {
            'total_items': total_items,
            'correct_items': correct_items,
            'error_items': error_items,
            'accuracy': correct_items / total_items if total_items > 0 else 0,
            'error_rate': error_items / total_items if total_items > 0 else 0,
            'english_stats': {
                'total': len(english_results),
                'correct': english_correct,
                'accuracy': english_correct / len(english_results) if english_results else 0
            },
            'chinese_stats': {
                'total': len(chinese_results),
                'correct': chinese_correct,
                'accuracy': chinese_correct / len(chinese_results) if chinese_results else 0
            },
            'model_name': self.model_name
        }

        return stats

    def save_results(self, results: List[Dict[str, Any]], file_path: str):
        """
        Save evaluation results to file

        Args:
            results: Evaluation results
            file_path: Output file path
        """
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Prepare output data
        output_data = {
            'model_name': self.model_name,
            'evaluation_timestamp': str(asyncio.get_event_loop().time()),
            'statistics': self.calculate_statistics(results),
            'results': results
        }

        # Save as JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_socialbench_data(data_path: str) -> List[Dict[str, Any]]:
        """
        Load SocialBench dataset from file

        Args:
            data_path: Path to dataset file

        Returns:
            List of test items
        """
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        if data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif data_path.suffix == '.jsonl':
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    def create_evaluation_report(self, results: List[Dict[str, Any]],
                               output_path: str) -> str:
        """
        Create detailed evaluation report

        Args:
            results: Evaluation results
            output_path: Path to save report

        Returns:
            Report content
        """
        stats = self.calculate_statistics(results)

        report = f"""
# SocialBench Evaluation Report

## Model Information
- **Model Name**: {self.model_name}
- **Evaluation Date**: {asyncio.get_event_loop().time()}

## Overall Performance
- **Total Items**: {stats['total_items']}
- **Correct Items**: {stats['correct_items']}
- **Accuracy**: {stats['accuracy']:.2%}
- **Error Rate**: {stats['error_rate']:.2%}

## Language-Specific Performance

### English
- **Total Items**: {stats['english_stats']['total']}
- **Correct Items**: {stats['english_stats']['correct']}
- **Accuracy**: {stats['english_stats']['accuracy']:.2%}

### Chinese
- **Total Items**: {stats['chinese_stats']['total']}
- **Correct Items**: {stats['chinese_stats']['correct']}
- **Accuracy**: {stats['chinese_stats']['accuracy']:.2%}

## Error Analysis
{len([r for r in results if 'error' in r])} items encountered errors during evaluation.

## Recommendations
{'✅ Performance is excellent!' if stats['accuracy'] > 0.8 else '⚠️ Performance could be improved.'}

---

*Report generated by R-CHAR SocialBench Evaluator*
        """

        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report.strip())

        return report.strip()


__all__ = [
    'SocialBenchEvaluator'
]