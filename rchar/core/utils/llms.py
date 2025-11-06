"""
LLM Interface Utilities

Provides a unified interface for interacting with various LLM backends
including OpenAI-compatible APIs and local models.
"""

import asyncio
import json
from typing import Optional, Dict, Any, List
from openai import AsyncClient
import json_repair


class JSONParseError(Exception):
    """Exception raised when JSON parsing fails"""
    def __init__(self, response: str, prompt: str):
        self.response = response
        self.prompt = prompt
        super().__init__(f"JSONParseError: {self.response[:100]}")


def create_llm_client(base_url: str = "https://api.openai.com/v1/",
                     api_key: str = "EMPTY",
                     timeout: float = 60.0) -> AsyncClient:
    """
    Create an LLM client instance

    Args:
        base_url: API base URL
        api_key: API key
        timeout: Request timeout in seconds

    Returns:
        AsyncClient instance
    """
    return AsyncClient(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout
    )


async def ask(prompt: str,
              model: str,
              llm_client: AsyncClient,
              system: Optional[str] = None,
              temperature: float = 0.7,
              max_tokens: Optional[int] = None,
              timeout: Optional[float] = None,
              json_mode: bool = False,
              json_list_mode: bool = False) -> str:
    """
    Send a prompt to LLM and get response

    Args:
        prompt: User prompt
        model: Model name
        llm_client: LLM client instance
        system: System prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout
        json_mode: Force JSON response format
        json_list_mode: Force JSON list response format

    Returns:
        LLM response string
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }

    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    if timeout:
        kwargs["timeout"] = timeout

    if json_mode or json_list_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = await llm_client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip()

        # Validate JSON if requested
        if json_mode or json_list_mode:
            try:
                parsed = json.loads(content)
                if json_list_mode and not isinstance(parsed, list):
                    raise ValueError("Expected JSON list but got object")
                return content
            except json.JSONDecodeError:
                # Try to repair JSON
                repaired = json_repair.repair_json(content)
                try:
                    json.loads(repaired)
                    return repaired
                except json.JSONDecodeError:
                    raise JSONParseError(content, prompt)

        return content

    except Exception as e:
        raise RuntimeError(f"LLM API call failed: {e}")


async def ask_json(prompt: str,
                  model: str,
                  llm_client: AsyncClient,
                  system: Optional[str] = None,
                  temperature: float = 0.7,
                  max_tokens: Optional[int] = None,
                  timeout: Optional[float] = None) -> Dict[str, Any]:
    """
    Send a prompt to LLM and get JSON response

    Args:
        prompt: User prompt
        model: Model name
        llm_client: LLM client instance
        system: System prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout

    Returns:
        Parsed JSON response as dictionary
    """
    if not system:
        system = "You must respond with valid JSON only."

    response = await ask(
        prompt=prompt,
        model=model,
        llm_client=llm_client,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        json_mode=True
    )

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        raise JSONParseError(response, prompt)


async def ask_json_list(prompt: str,
                       model: str,
                       llm_client: AsyncClient,
                       system: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       timeout: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Send a prompt to LLM and get JSON list response

    Args:
        prompt: User prompt
        model: Model name
        llm_client: LLM client instance
        system: System prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout

    Returns:
        Parsed JSON response as list
    """
    if not system:
        system = "You must respond with a valid JSON array only."

    response = await ask(
        prompt=prompt,
        model=model,
        llm_client=llm_client,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        json_list_mode=True
    )

    try:
        result = json.loads(response)
        if not isinstance(result, list):
            raise ValueError("Expected JSON list but got object")
        return result
    except json.JSONDecodeError as e:
        raise JSONParseError(response, prompt)


async def ask_messages(messages: List[Dict[str, str]],
                      model: str,
                      llm_client: AsyncClient,
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None,
                      timeout: Optional[float] = None) -> str:
    """
    Send message list to LLM and get response

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model name
        llm_client: LLM client instance
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout

    Returns:
        LLM response string
    """
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }

    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    if timeout:
        kwargs["timeout"] = timeout

    try:
        response = await llm_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"LLM API call failed: {e}")


# Pre-configured client factories for common use cases
def create_openai_client(api_key: str, base_url: str = "https://api.openai.com/v1/") -> AsyncClient:
    """Create OpenAI client"""
    return create_llm_client(base_url=base_url, api_key=api_key)


def create_ollama_client(base_url: str = "http://localhost:11434/v1/") -> AsyncClient:
    """Create Ollama client for local models"""
    return create_llm_client(base_url=base_url, api_key="EMPTY")


def create_azure_client(api_key: str, base_url: str, api_version: str = "2023-12-01-preview") -> AsyncClient:
    """Create Azure OpenAI client"""
    return create_llm_client(base_url=base_url, api_key=api_key)


# Common model configurations
COMMON_MODELS = {
    "gpt-4": "gpt-4",
    "gpt-4-turbo": "gpt-4-1106-preview",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "qwen2.5-32b": "qwen2.5:32b",
    "qwen2.5-72b": "qwen2.5:72b"
}


__all__ = [
    'ask', 'ask_json', 'ask_json_list', 'ask_messages',
    'create_llm_client', 'create_openai_client', 'create_ollama_client', 'create_azure_client',
    'JSONParseError', 'COMMON_MODELS'
]