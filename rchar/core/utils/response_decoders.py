"""
Response Decoder Utilities

Utilities for parsing and extracting structured information from LLM responses,
including XML tag extraction and key-value parsing.
"""

import re
from typing import Dict, Any, List, Optional


def extract_tags_dict(response: str, tags: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Extract content from XML-like tags in LLM response

    Args:
        response: LLM response text
        tags: List of tags to extract. If None, extracts all tags found.

    Returns:
        Dictionary mapping tag names to their content
    """
    if tags is None:
        # Auto-detect all tags
        tag_pattern = r'<(\w+)>'
        found_tags = re.findall(tag_pattern, response)
        tags = list(set(found_tags))

    result = {}

    for tag in tags:
        # Pattern to match content between opening and closing tags
        pattern = f'<{tag}>(.*?)</{tag}>'
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            # Take the first match for each tag
            result[tag] = matches[0].strip()
        else:
            # Try alternative pattern with self-closing or missing closing tag
            alt_pattern = f'<{tag}>(.*?)(?=<\w+>|$)'
            alt_matches = re.findall(alt_pattern, response, re.DOTALL)
            if alt_matches:
                result[tag] = alt_matches[0].strip()
            else:
                result[tag] = ""

    return result


def extract_key_values(text: str, delimiter: str = ":") -> Dict[str, str]:
    """
    Extract key-value pairs from text

    Args:
        text: Text containing key-value pairs
        delimiter: Delimiter between keys and values

    Returns:
        Dictionary of extracted key-value pairs
    """
    result = {}

    # Pattern to match key: value pairs, handling multi-line values
    pattern = rf'(.+?)\s*{delimiter}\s*(.*?)(?=\n\s*\w+\s*{delimiter}|\n\s*$|$)'

    matches = re.findall(pattern, text, re.DOTALL)

    for key, value in matches:
        # Clean up key and value
        key = key.strip().strip('-*#').strip()
        value = value.strip()

        # Remove list markers and clean up formatting
        value = re.sub(r'^[\s\-\*\#]+', '', value, flags=re.MULTILINE)
        value = re.sub(r'\n\s*', ' ', value)

        if key and value:
            result[key] = value

    return result


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from text

    Args:
        text: Text to extract numbers from

    Returns:
        List of numbers found
    """
    # Pattern to match integers and floats
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches]


def extract_score(text: str, score_range: tuple = (0, 10)) -> Optional[float]:
    """
    Extract a score within specified range from text

    Args:
        text: Text containing score
        score_range: Valid range for score (min, max)

    Returns:
        Extracted score or None if not found/valid
    """
    numbers = extract_numbers(text)

    for num in numbers:
        if score_range[0] <= num <= score_range[1]:
            return num

    return None


def extract_list_items(text: str, markers: Optional[List[str]] = None) -> List[str]:
    """
    Extract list items from text

    Args:
        text: Text containing list items
        markers: List markers to look for (default: common bullet points)

    Returns:
        List of extracted items
    """
    if markers is None:
        markers = ['-', '*', '•', '1.', '2.', '3.', 'a.', 'b.', 'c.']

    # Create pattern for list markers
    marker_pattern = '|'.join(re.escape(marker) for marker in markers)
    pattern = rf'^\s*(?:{marker_pattern})\s*(.+)$'

    matches = re.findall(pattern, text, re.MULTILINE)

    # Clean up matches
    items = []
    for match in matches:
        cleaned = match.strip()
        if cleaned:
            items.append(cleaned)

    return items


def clean_text(text: str) -> str:
    """
    Clean and normalize text

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Remove common formatting artifacts
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove empty lines
    text = re.sub(r'\s*\n\s*', ' ', text)  # Replace line breaks with spaces

    return text


def parse_json_fallback(text: str) -> Dict[str, Any]:
    """
    Attempt to parse JSON with fallback methods

    Args:
        text: Text potentially containing JSON

    Returns:
        Parsed JSON or empty dict if parsing fails
    """
    import json

    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from text
    json_pattern = r'\{.*\}|\[.*\]'
    matches = re.findall(json_pattern, text, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Return empty dict if all attempts fail
    return {}


def extract_rating(text: str, scale: int = 10) -> Optional[int]:
    """
    Extract rating/score from text on specified scale

    Args:
        text: Text containing rating
        scale: Maximum rating value

    Returns:
        Extracted rating as integer or None
    """
    # Look for patterns like "7/10", "7 out of 10", "Rating: 7"
    patterns = [
        rf'(\d+)/{scale}',
        rf'(\d+)\s*out\s*of\s*{scale}',
        rf'rating\s*[:\-]?\s*(\d+)',
        rf'score\s*[:\-]?\s*(\d+)'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                return int(matches[0])
            except ValueError:
                continue

    # Try to extract any number in range
    numbers = extract_numbers(text)
    for num in numbers:
        if 0 <= num <= scale:
            return int(num)

    return None


__all__ = [
    'extract_tags_dict',
    'extract_key_values',
    'extract_numbers',
    'extract_score',
    'extract_list_items',
    'clean_text',
    'parse_json_fallback',
    'extract_rating'
]