# R-CHAR: Role-Consistent Hierarchical Adaptive Reasoning Framework

This repository contains the implementation of R-CHAR (Role-Consistent Hierarchical Adaptive Reasoning), a framework for enhancing role-playing performance in large language models through metacognition-driven thinking trajectory optimization.

## Overview

R-CHAR addresses a fundamental challenge in AI role-playing: achieving authentic character embodiment rather than superficial mimicry. Traditional approaches often focus on language style matching, but our framework introduces explicit thinking trajectory guidance to enable deeper cognitive consistency.

The core innovation is extending high-quality thinking processes through continuation prompts. We discovered that higher-scoring responses consistently feature longer, more in-depth thinking trajectories. R-CHAR leverages this insight by using `(CONTINUE YOUR THINKING...)` prompts to prolong and deepen already successful thinking processes:

```
</think>
[Character's internal reasoning and analysis]
(CONTINUE YOUR THINKING...)
[Additional deep thinking and refinement]
<think>

<answer>
[Final character response based on the thinking process]
</answer>
```

## Installation

```bash
git clone https://github.com/lavapapa/rchar.git
cd rchar
pip install -r requirements.txt
```

## Quick Start

### Environment Setup

```bash
# Python 3.8+ required
python --version

# Run local model service
# Use vllm or sglang to serve your model

# Configure API keys for cloud services
# export OPENAI_API_KEY="your-key"
```

### Running Examples

```bash
# Basic optimization example
python examples/basic_usage.py

# Evaluation example
python examples/evaluation_example.py
```
