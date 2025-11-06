# Contributing to R-CHAR Framework

Thank you for your interest in contributing to the R-CHAR (Role-Consistent Hierarchical Adaptive Reasoning) framework! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

We use GitHub Issues to track public bugs and feature requests. Please:

1. **Search existing issues** before creating a new one
2. **Use clear, descriptive titles** for issues
3. **Provide detailed information** including:
   - Your environment (Python version, OS, etc.)
   - Steps to reproduce the issue
   - Expected vs. actual behavior
   - Any error messages or logs

### Submitting Pull Requests

1. **Fork the repository** and create a new branch for your feature
2. **Follow the coding standards** outlined below
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass** before submitting

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Setup Steps

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/rchar.git
cd rchar

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in development mode
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Create a new branch for your feature
git checkout -b feature/your-feature-name
```

## 📝 Coding Standards

### Code Style

We use `black` for code formatting and `ruff` for linting:

```bash
# Format code
black rchar/ examples/ tests/

# Lint code
ruff check rchar/ examples/ tests/

# Fix linting issues
ruff check --fix rchar/ examples/ tests/
```

### Naming Conventions

- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Documentation

- **Docstrings**: Use Google-style docstrings
- **Type hints**: Include type hints for all public functions
- **Comments**: Explain complex logic, not obvious code

Example:

```python
def optimize_roleplay(
    persona: str,
    scenario: str,
    llm_config: LLMConfig,
    max_iterations: int = 4
) -> OptimizationResult:
    """Optimize role-playing performance using R-CHAR framework.

    Args:
        persona: Character description and background
        scenario: Role-playing scenario context
        llm_config: Configuration for LLM clients
        max_iterations: Maximum optimization iterations

    Returns:
        OptimizationResult containing the optimized response and metadata

    Raises:
        ValueError: If persona or scenario is empty
        LLMError: If LLM API calls fail
    """
    pass
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rchar --cov-report=html

# Run specific test file
pytest tests/test_core_engine.py

# Run with verbose output
pytest -v
```

### Writing Tests

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **Use fixtures**: For common test setup
- **Mock external dependencies**: LLM APIs, file systems, etc.

Example test:

```python
import pytest
from unittest.mock import AsyncMock, patch
from rchar.core.core_engine import RCharengine

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = AsyncMock()
    client.chat.completions.create.return_value.choices[0].message.content = "Test response"
    return client

@pytest.mark.asyncio
async def test_generate_scenario(mock_llm_client):
    """Test scenario generation functionality."""
    engine = RCharengine(debug_mode=True)

    scenarios = await engine.generate_scenario(
        persona="Test character",
        llm_client=mock_llm_client,
        model="test-model"
    )

    assert len(scenarios) > 0
    assert scenarios[0].character == "Test character"
```

## 📚 Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and inline comments
2. **User Documentation**: README, examples, tutorials
3. **API Documentation**: Generated from docstrings
4. **Developer Documentation**: Architecture guides, contributing guidelines

### Documentation Standards

- **Keep it up-to-date** with code changes
- **Use clear, simple language**
- **Include code examples** where helpful
- **Add diagrams** for complex concepts (using Mermaid)

## 🏗️ Architecture Guidelines

### Project Structure

- **Core functionality**: `rchar/core/`
- **Evaluation tools**: `rchar/evaluation/`
- **Dataset handling**: `rchar/datasets/`
- **Examples**: `examples/`
- **Tests**: `tests/`

### Design Principles

1. **Modularity**: Keep components loosely coupled
2. **Testability**: Design for easy testing
3. **Extensibility**: Make it easy to add new features
4. **Performance**: Use async/await for I/O operations
5. **Type Safety**: Use type hints throughout

### Adding New Features

1. **Design the API** first (interfaces, type hints)
2. **Implement the functionality**
3. **Add comprehensive tests**
4. **Write documentation**
5. **Update examples** if relevant

## 🔄 Development Workflow

### Before Submitting

1. **Run all tests**: `pytest`
2. **Check code style**: `black` and `ruff`
3. **Update documentation**
4. **Add/update tests** for new functionality
5. **Verify examples** still work

### Pull Request Process

1. **Create a descriptive title** for your PR
2. **Describe the changes** in the description
3. **Link relevant issues**
4. **Add screenshots** if applicable
5. **Request code review** from maintainers

### Code Review Guidelines

- **Be constructive** and respectful
- **Focus on code quality**, not personal preferences
- **Suggest improvements** where appropriate
- **Ask questions** if anything is unclear

## 🐛 Bug Fixes

### Fixing Bugs

1. **Add a test** that reproduces the bug
2. **Fix the issue** with minimal changes
3. **Verify the fix** with the new test
4. **Check for regressions** in existing functionality
5. **Document the change** if needed

### Bug Report Template

```markdown
## Bug Description
[Clear description of the issue]

## Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Environment
- OS: [Your OS]
- Python version: [Your Python version]
- R-CHAR version: [Version]

## Additional Context
[Any other relevant information]
```

## ✨ Feature Requests

### Requesting Features

1. **Check existing issues** and roadmaps
2. **Describe the use case** clearly
3. **Explain why it's valuable**
4. **Consider implementation** complexity
5. **Be open to discussion** and feedback

### Feature Request Template

```markdown
## Feature Description
[Clear description of the proposed feature]

## Problem Statement
[What problem does this solve?]

## Proposed Solution
[How should this work?]

## Use Cases
[Who would use this and how?]

## Alternatives Considered
[What other approaches did you consider?]

## Additional Notes
[Any other relevant information]
```

## 🎯 Priority Areas

We welcome contributions in these areas:

- **New optimization methods** and thinking trajectory strategies
- **Additional evaluation metrics** and benchmarks
- **Performance improvements** and optimizations
- **Documentation** improvements and tutorials
- **Integration with more LLM providers**
- **Domain-specific optimizations** (education, entertainment, etc.)

## 📞 Getting Help

- **GitHub Discussions**: For general questions and ideas
- **GitHub Issues**: For bug reports and feature requests
- **Discord/Slack**: [Link to community chat] (if available)
- **Email**: [Contact email] (for maintainers only)

## 🙏 Recognition

Contributors will be recognized in:

- **README.md**: Top contributors section
- **CHANGELOG.md**: For each release
- **Contributors.md**: Detailed contributor list
- **Release notes**: For significant contributions

---

Thank you for contributing to R-CHAR! Your contributions help make this project better for everyone. 🚀