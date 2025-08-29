#!/usr/bin/env python3
"""
Script to list all non-trivial (non-wrapper) Rust methods in the FFI layer.
Scans all .rs files in python-groggy/src/ffi and prints a list of method names with their file and line number.

Non-trivial methods are heuristically detected as those whose body is not a single delegation or trivial return.
This script does not do full Rust parsing, but uses simple heuristics:
- Finds all function definitions (fn ...)
- For each, checks if the body is more than a single statement or contains control flow (if, for, while, match)
- Prints file, method name, and line number

Usage:
    python scripts/list_nontrivial_ffi_methods.py
"""
import os

import os
import re

FFI_DIR = os.path.join(os.path.dirname(__file__), '..', 'python-groggy', 'src', 'ffi')
FUNC_DEF_RE = re.compile(r'^(\s*(pub\s+)?(async\s+)?fn\s+([a-zA-Z0-9_]+))\s*\(')
LOOP_RE = re.compile(r'\b(for|while|loop)\b')
CALL_RE = re.compile(r'([a-zA-Z0-9_]+)\s*\(')
FFI_DIR = os.path.join(os.path.dirname(__file__), '..', 'python-groggy', 'src', 'ffi')
FUNC_DEF_RE = re.compile(r'^(\s*(pub\s+)?(async\s+)?fn\s+([a-zA-Z0-9_]+))\s*\(')
LOOP_RE = re.compile(r'\b(for|while|loop)\b')
CALL_RE = re.compile(r'([a-zA-Z0-9_]+)\s*\(')


def list_rs_files(root):
    # Only include .rs files in ffi/ and ffi/api/, not ffi/core or other subfolders
    allowed_dirs = {root, os.path.join(root, 'api')}
    for dirpath, _, filenames in os.walk(root):
        if dirpath not in allowed_dirs:
            continue
        for fname in filenames:
            if fname.endswith('.rs'):
                yield os.path.join(dirpath, fname)

def extract_methods(filepath):
    methods = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        m = FUNC_DEF_RE.match(line)
        if m:
            name = m.group(4)
            # Find function body (naive: look for next '{' and match braces)
            body_start = None
            for j in range(i, min(i+10, len(lines))):
                if '{' in lines[j]:
                    body_start = j
                    break
            if body_start is None:
                continue
            brace_count = 0
            body_lines = []
            for k in range(body_start, len(lines)):
                brace_count += lines[k].count('{')
                brace_count -= lines[k].count('}')
                body_lines.append(lines[k])
                if brace_count == 0:
                    break
            body = ''.join(body_lines)
            # Heuristic 1: Contains a loop
            if LOOP_RE.search(body):
                methods.append({'file': filepath, 'line': i+1, 'method': name, 'reason': 'loop'})
                continue
            # Heuristic 2: More than one function/method call (excluding self/this/field access)
            calls = [c for c in CALL_RE.findall(body) if c not in {'if', 'for', 'while', 'loop', 'match', 'return', 'Some', 'Ok', 'Err', 'None', 'self', 'Self'}]
            if len(calls) > 2:
                methods.append({'file': filepath, 'line': i+1, 'method': name, 'reason': f'{len(calls)} calls'})
    return methods

def main():
    all_methods = []
    for rsfile in list_rs_files(FFI_DIR):
        all_methods.extend(extract_methods(rsfile))
    for m in all_methods:
        print(f"{os.path.relpath(m['file'], FFI_DIR)}:{m['line']}: {m['method']}  # {m['reason']}")

if __name__ == '__main__':
    main()
