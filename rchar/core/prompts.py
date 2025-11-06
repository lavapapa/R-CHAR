"""
Prompt Templates for R-CHAR Framework

This module contains all prompt templates used in the R-CHAR framework,
including role-playing, scenario generation, evaluation, and introspection prompts.
"""

# Role-playing system prompt
ROLEPLAY_SYSTEM_PROMPT = """You are a Roleplay Assistant. You will play the role of a character in a given scenario.

Before responding to the instruction, think step by step in <think>...</think> and then respond in <answer>...</answer>."""

# Role-playing user prompt template
ROLEPLAY_USER_PROMPT = """{character}
{scenario}
{instruction}
"""

# Role-playing user prompt with evaluation criteria
ROLEPLAY_USER_PROMPT_WITH_CRITERIA = """{character}
{scenario}

# Evaluation Criteria
{criteria}

# Answer the following instruction
{instruction}
"""

# Persona to scenario conversion prompt
PERSONA_TO_SCENARIO_PROMPT = """You are a professional character designer and scriptwriter. Your task is to create three scenarios at different difficulty levels to evaluate roleplay quality for a given character.

# Task Description
Create three scenarios and instructions at different difficulty levels (Basic/Advanced/Expert), each designed to evaluate specific roleplay capabilities while ensuring consistency with the character's background and world setting.

# Input Character
<character>
{persona}
</character>

# Output in JSON Format:
{{
    "analysis": "analysis the language patterns, personality traits, knowledge boundaries, and core values of the character, briefly",
    "scenarios": [
        {{
            "level": "basic/advanced/expert",
            "scenario": "A scenario testing roleplay capabilities",
            "instruction": "Ask the character to respond to the scenario",
        }}, ...
    ]
}}

# Difficulty Level Requirements
- Basic Level:
Examine basic language patterns, behavioral modes, character consistency through simple scenarios while avoiding hallucinations and testing fundamental character settings

- Advanced Level:
Examine emotional expression, interpersonal interactions, multilayered decision-making, knowledge boundaries, and professional performance within character identity

- Difficult Level:
Examine character's decision-making under extreme circumstances involving value conflicts, demonstrating complex internal reasoning and growth while maintaining authenticity

# General Requirements
- All scenarios should feel natural, not forced
- Challenges should be meaningful but not impossible
- Instructions should be specific but open enough for creative responses
- Consider both external and internal conflicts
- Ensure all elements respect the character's established background

Now, please analyze the character and generate appropriate scenarios and instructions for all three difficulty levels in JSON format.
"""

# Criteria generation prompt
CRITERIA_GENERATE_PROMPT = """You are an expert evaluator specializing in character performance assessment. Your task is to create comprehensive evaluation criteria for role-playing scenarios.

# Task Description
Generate specific, measurable evaluation criteria for assessing the quality of character role-playing in the given scenario.

# Character Information
<character>
{character}
</character>

# Scenario Context
<scenario>
{scenario}
</scenario>

# Role-playing Instruction
<instruction>
{instruction}
</instruction>

# Output Format
<criteria_analysis>
Brief analysis of what makes this role-playing challenging and what key aspects should be evaluated
</criteria_analysis>

<evaluation_criteria>
Provide 5-7 specific criteria, each with:
1. A clear criterion name
2. Description of what excellent performance looks like
3. Rating scale (1-10)

Format each criterion as:
**Criterion Name**: [Description - Excellent performance shows X, Y, Z]
</evaluation_criteria>

# Focus Areas
Consider evaluating:
- Character consistency and authenticity
- Emotional expression appropriateness
- Knowledge boundary adherence
- Decision-making quality
- Language style matching
- Response relevance and coherence
- Depth of character engagement

Please generate comprehensive evaluation criteria for this role-playing scenario.
"""

# Performance evaluation prompt
EVALUATE_ANALYSIS_PROMPT = """You are an expert evaluator specializing in character performance assessment. Your task is to evaluate a character's role-playing performance based on specific criteria.

# Character Information
<character>
{character}
</character>

# Scenario Context
<scenario>
{scenario}
</scenario>

# Role-playing Instruction
<instruction>
{instruction}
</instruction>

# Evaluation Criteria
<criteria>
{criteria}
</criteria>

# Character's Performance
<thinking_process>
{think}
</thinking_process>

<response>
{answer}
</response>

# Evaluation Task
Please evaluate the character's performance based on the provided criteria.

# Output Format
<overall_assessment>
Provide an overall assessment of the performance, including strengths and areas for improvement
</overall_assessment>

<criteria_evaluation>
Evaluate each criterion from the provided criteria list. For each criterion:
1. Provide a score from 1-10
2. Justify the score with specific examples from the performance

Format as:
**Criterion Name**: [Score]/10 - [Specific justification with examples]
</criteria_evaluation>

<overall_flaws>
Identify the main weaknesses or issues in the performance that need improvement
</overall_flaws>

<specific_suggestions>
Provide concrete suggestions for improving the performance
</specific_suggestions>

Please provide a comprehensive evaluation of this role-playing performance.
"""

# Introspective guidance generation prompt
GENERATE_INTROSPECTIVE_PROMPT = """You are an expert acting coach specializing in character development and role-playing improvement. Your task is to provide introspective guidance to help the character improve their performance.

# Character Information
<character>
{character}
</character>

# Scenario Context
<scenario>
{scenario}
</scenario>

# Role-playing Instruction
<instruction>
{instruction}
</instruction>

# Evaluation Criteria
<criteria>
{criteria}
</criteria>

# Current Performance Analysis
<thinking_process>
{think}
</thinking_process>

<performance_evaluation>
<overall_flaws>
{overall_flaws}
</overall_flaws>

<criteria_scores>
{criteria_evaluation}
</criteria_scores>
</performance_evaluation>

# Your Task
Provide specific, actionable guidance to help the character improve their thinking process and performance in this scenario.

# Output Format
<guidance>
Provide introspective guidance that helps the character:
1. Understand their current performance limitations
2. Develop better thinking strategies
3. Improve their character authenticity
4. Make more appropriate decisions
5. Enhance their emotional expression

The guidance should be:
- Specific and actionable
- Character-focused (not generic advice)
- Constructive and encouraging
- Address the identified flaws
- Help develop better internal reasoning

Frame the guidance as if you're coaching the character on how to think and respond more effectively while staying true to their character.
</guidance>

Please provide detailed introspective guidance for performance improvement.
"""

# Alternative prompts for ablation studies

# Reference optimization prompt (for ablation experiments)
REFERENCE_OPTIMIZATION_PROMPT = """You are an expert role-playing coach. Your task is to improve a character's response based on a reference example.

# Character Information
<character>
{character}
</character>

# Scenario Context
<scenario>
{scenario}
</scenario>

# Role-playing Instruction
<instruction>
{instruction}
</instruction>

# Current Response
<current_response>
{answer}
</current_response>

# Reference Response (Example of Excellent Performance)
<reference_response>
{reference_answer}
</reference_response>

# Your Task
Analyze the reference response and create an improved version of the current response that:
1. Maintains the character's authentic voice
2. Demonstrates better understanding of the scenario
3. Shows more appropriate emotional responses
4. Makes more character-consistent decisions
5. Incorporates the best qualities from the reference response

# Output Format
<improved_response>
[Your improved character response]
</improved_response>

<improvement_explanation>
Explain how your response improves upon the original while maintaining character authenticity
</improvement_explanation>

Please provide an improved response that bridges the gap between the current performance and the reference example.
"""

# Simple selection prompt (for ablation experiments)
SIMPLE_SELECTION_PROMPT = """You are evaluating multiple role-playing responses for the same character and scenario.

# Character Information
<character>
{character}
</character>

# Scenario Context
<scenario>
{scenario}
</scenario>

# Role-playing Instruction
<instruction>
{instruction}
</instruction>

# Response Options
{responses}

# Your Task
Select the best response based on:
1. Character consistency and authenticity
2. Emotional appropriateness
3. Decision-making quality
4. Language style matching
5. Overall engagement with the scenario

# Output Format
<best_response>
[Copy the best response exactly as provided]
</best_response>

<selection_reasoning>
Explain why this response is the best choice, citing specific examples
</selection_reasoning>

<response_ranking>
Rank all responses from best to worst with brief justifications
</response_ranking>

Please select the best response and provide your reasoning.
"""

# Rewrite and optimization prompt
REWRITE_OPTIMIZATION_PROMPT = """You are an expert editor specializing in character dialogue and thinking processes. Your task is to rewrite and optimize a character's thinking trajectory to improve clarity, consistency, and depth.

# Character Information
<character>
{character}
</character>

# Scenario Context
<scenario>
{scenario}
</scenario>

# Role-playing Instruction
<instruction>
{instruction}
</instruction>

# Original Thinking Process
<thinking_process>
{think}
</thinking_process>

# Original Response
<response>
{answer}
</response>

# Your Task
Rewrite the thinking process to:
1. Improve logical flow and coherence
2. Enhance character authenticity
3. Deepen emotional engagement
4. Strengthen decision-making rationale
5. Eliminate inconsistencies or contradictions
6. Make connections more explicit
7. Ensure the thinking naturally leads to the response

# Output Format
<rewritten_thinking>
[Your optimized thinking process in <think>...</think> format]
</rewritten_thinking>

<rewritten_response>
[Corresponding response that matches the rewritten thinking]
</rewritten_response>

<optimization_explanation>
Explain the key improvements you made and why they enhance the character's performance
</optimization_explanation>

Please rewrite the thinking process to create a more coherent and authentic character performance.
"""

# Export all prompts
__all__ = [
    'ROLEPLAY_SYSTEM_PROMPT',
    'ROLEPLAY_USER_PROMPT',
    'ROLEPLAY_USER_PROMPT_WITH_CRITERIA',
    'PERSONA_TO_SCENARIO_PROMPT',
    'CRITERIA_GENERATE_PROMPT',
    'EVALUATE_ANALYSIS_PROMPT',
    'GENERATE_INTROSPECTIVE_PROMPT',
    'REFERENCE_OPTIMIZATION_PROMPT',
    'SIMPLE_SELECTION_PROMPT',
    'REWRITE_OPTIMIZATION_PROMPT'
]