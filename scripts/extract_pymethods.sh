#!/bin/bash
# Extract all #[pymethods] and #[pyfunction] blocks and their method signatures from Rust FFI source files
# Usage: bash extract_pymethods.sh > pymethods_report.txt

FFI_DIR="python-groggy/src/ffi"

# Find all relevant Rust files in the FFI directory
grep -r -n -E '#\[pymethods\]|#\[pyfunction\]' "$FFI_DIR" | while read -r line; do
    file=$(echo "$line" | cut -d: -f1)
    lineno=$(echo "$line" | cut -d: -f2)
    marker=$(echo "$line" | cut -d: -f3-)
    echo "\n$file:$lineno $marker"
    # Print the next 30 lines after the marker for context
    awk -v start=$((lineno+1)) -v end=$((lineno+30)) 'NR>=start && NR<=end' "$file"
done | grep -E '(^[a-zA-Z0-9_ ]*fn |impl |struct |#\[pymethods\]|#\[pyfunction\]|^$)'
