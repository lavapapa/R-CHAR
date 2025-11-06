"""
JSONL File Utilities

Utilities for reading and writing JSONL (JSON Lines) files,
which are commonly used for large-scale dataset processing.
"""

import json
import os
from typing import List, Dict, Any, Iterator, Optional, Union
from pathlib import Path


def read_jsonl(file_path: Union[str, Path], encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """
    Read all lines from a JSONL file

    Args:
        file_path: Path to JSONL file
        encoding: File encoding

    Returns:
        List of dictionaries parsed from JSONL file
    """
    data = []
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    with open(file_path, 'r', encoding=encoding) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON decode error at line {line_num} in {file_path}: {e}")

    return data


def read_jsonl_stream(file_path: Union[str, Path], encoding: str = 'utf-8') -> Iterator[Dict[str, Any]]:
    """
    Stream read from a JSONL file (memory efficient for large files)

    Args:
        file_path: Path to JSONL file
        encoding: File encoding

    Yields:
        Dictionary for each line in JSONL file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    with open(file_path, 'r', encoding=encoding) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON decode error at line {line_num} in {file_path}: {e}")


def write_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path],
                encoding: str = 'utf-8', ensure_ascii: bool = False, indent: Optional[int] = None) -> None:
    """
    Write data to a JSONL file

    Args:
        data: List of dictionaries to write
        file_path: Output file path
        encoding: File encoding
        ensure_ascii: Whether to ensure ASCII encoding
        indent: Optional indentation for pretty printing (not recommended for JSONL)
    """
    file_path = Path(file_path)

    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding=encoding) as f:
        for item in data:
            if indent:
                # Pretty print (not standard JSONL format)
                json_str = json.dumps(item, ensure_ascii=ensure_ascii, indent=indent)
            else:
                # Standard JSONL format
                json_str = json.dumps(item, ensure_ascii=ensure_ascii)
            f.write(json_str + '\n')


def append_jsonl(item: Dict[str, Any], file_path: Union[str, Path],
                 encoding: str = 'utf-8', ensure_ascii: bool = False) -> None:
    """
    Append a single item to a JSONL file

    Args:
        item: Dictionary to append
        file_path: Output file path
        encoding: File encoding
        ensure_ascii: Whether to ensure ASCII encoding
    """
    file_path = Path(file_path)

    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'a', encoding=encoding) as f:
        json_str = json.dumps(item, ensure_ascii=ensure_ascii)
        f.write(json_str + '\n')


def merge_jsonl_files(input_paths: List[Union[str, Path]], output_path: Union[str, Path],
                      encoding: str = 'utf-8', deduplicate: bool = False,
                      deduplicate_key: Optional[str] = None) -> int:
    """
    Merge multiple JSONL files into one

    Args:
        input_paths: List of input JSONL file paths
        output_path: Output file path
        encoding: File encoding
        deduplicate: Whether to remove duplicates
        deduplicate_key: Key to use for deduplication (required if deduplicate=True)

    Returns:
        Number of items written to output file
    """
    if deduplicate and not deduplicate_key:
        raise ValueError("deduplicate_key is required when deduplicate=True")

    seen_keys = set() if deduplicate else None
    total_items = 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding=encoding) as out_file:
        for input_path in input_paths:
            input_path = Path(input_path)
            if not input_path.exists():
                print(f"⚠️ Warning: File not found, skipping: {input_path}")
                continue

            for item in read_jsonl_stream(input_path, encoding):
                if deduplicate:
                    key = item.get(deduplicate_key)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                json_str = json.dumps(item, ensure_ascii=False)
                out_file.write(json_str + '\n')
                total_items += 1

    return total_items


def filter_jsonl(input_path: Union[str, Path], output_path: Union[str, Path],
                 filter_func: callable, encoding: str = 'utf-8') -> int:
    """
    Filter JSONL file based on a predicate function

    Args:
        input_path: Input file path
        output_path: Output file path
        filter_func: Function that takes a dict and returns bool
        encoding: File encoding

    Returns:
        Number of items written to output file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, 'w', encoding=encoding) as out_file:
        for item in read_jsonl_stream(input_path, encoding):
            if filter_func(item):
                json_str = json.dumps(item, ensure_ascii=False)
                out_file.write(json_str + '\n')
                count += 1

    return count


def sample_jsonl(input_path: Union[str, Path], output_path: Union[str, Path],
                sample_size: int, random_seed: Optional[int] = None,
                encoding: str = 'utf-8') -> int:
    """
    Sample random items from JSONL file

    Args:
        input_path: Input file path
        output_path: Output file path
        sample_size: Number of items to sample
        random_seed: Random seed for reproducibility
        encoding: File encoding

    Returns:
        Number of items sampled
    """
    import random

    if random_seed is not None:
        random.seed(random_seed)

    # Read all items (could be memory intensive for large files)
    all_items = read_jsonl(input_path, encoding)

    if sample_size >= len(all_items):
        # If sample size is larger than file, just copy all
        write_jsonl(all_items, output_path, encoding)
        return len(all_items)

    # Random sample
    sampled_items = random.sample(all_items, sample_size)
    write_jsonl(sampled_items, output_path, encoding)

    return len(sampled_items)


def get_jsonl_stats(file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Get statistics about a JSONL file

    Args:
        file_path: Path to JSONL file
        encoding: File encoding

    Returns:
        Dictionary with file statistics
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    total_lines = 0
    valid_lines = 0
    file_size = file_path.stat().st_size

    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if line:
                try:
                    json.loads(line)
                    valid_lines += 1
                except json.JSONDecodeError:
                    pass

    return {
        'file_path': str(file_path),
        'file_size_bytes': file_size,
        'total_lines': total_lines,
        'valid_json_lines': valid_lines,
        'invalid_lines': total_lines - valid_lines,
        'validity_rate': valid_lines / total_lines if total_lines > 0 else 0
    }


def validate_jsonl(file_path: Union[str, Path], encoding: str = 'utf-8',
                   raise_on_error: bool = False) -> List[Dict[str, Any]]:
    """
    Validate JSONL file format

    Args:
        file_path: Path to JSONL file
        encoding: File encoding
        raise_on_error: Whether to raise exception on first error

    Returns:
        List of validation errors
    """
    errors = []
    file_path = Path(file_path)

    with open(file_path, 'r', encoding=encoding) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                error_info = {
                    'line_number': line_num,
                    'error': str(e),
                    'content': line[:100] + '...' if len(line) > 100 else line
                }
                errors.append(error_info)

                if raise_on_error:
                    raise ValueError(f"JSON validation error at line {line_num}: {e}")

    return errors


__all__ = [
    'read_jsonl',
    'read_jsonl_stream',
    'write_jsonl',
    'append_jsonl',
    'merge_jsonl_files',
    'filter_jsonl',
    'sample_jsonl',
    'get_jsonl_stats',
    'validate_jsonl'
]