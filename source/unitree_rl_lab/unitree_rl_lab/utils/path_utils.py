# -*- coding: utf-8 -*-
'''
@File    : path_utils.py
@Time    : 2026/05/02 13:03:21
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : Some utility for path operations.
'''

from pathlib import Path
from typing import Sequence
from itertools import chain


def resolve_files(files: str | Sequence[str], suffixes: str | Sequence[str]) -> list[str]:
    """Resolve file and directory inputs into a flat list of specified suffix files.

    Args:
        files: A single path or a sequence of file and directory paths.
        suffixes: The file suffix(es) to filter for.

    Returns:
        A sorted list of resolved file paths with the specified suffix.
    """
    entries = [files] if isinstance(files, str) else list(files)
    suffix_list = [suffixes] if isinstance(suffixes, str) else list(suffixes)
    suffix_list = [suffix if suffix.startswith(".") else f".{suffix}" for suffix in suffix_list]

    resolved_files: list[str] = []
    for entry in entries:
        path = Path(entry).expanduser().resolve()
        if path.is_dir():
            resolved_files.extend(str(file_path) for file_path in sorted(chain(*[path.rglob(f"*{suffix}") for suffix in suffix_list])) if file_path.is_file())
            continue
        if path.is_file():
            if path.suffix.lower() not in suffix_list:
                raise ValueError(f"File must be one of {suffix_list}, but got {path}")
            resolved_files.append(str(path))
            continue
        raise FileNotFoundError(f"File does not exist: {path}")

    resolved_files = list(dict.fromkeys(resolved_files))  # remove duplicates
    if not resolved_files:
        raise FileNotFoundError(f"No files with specified suffixes found in: {entries}")
    return resolved_files

if __name__ == '__main__':
    files = resolve_files([".vscode", "unitree_rl_lab"], [".md", ".json", ".npz"])
    files = resolve_files("unitree_rl_lab", ".npz")
    print(files)
