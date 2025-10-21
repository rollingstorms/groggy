#!/usr/bin/env python3
"""
Generate method documentation from the comprehensive test meta-graph.

Since Groggy classes are Rust (PyO3) and mkdocstrings can't read them,
we use the meta-graph to generate complete method listings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python-groggy" / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import groggy as gr
from comprehensive_library_testing import load_comprehensive_test_graph


def generate_method_table_for_object(graph, object_name):
    """Generate a markdown table of methods for an object."""

    # Get all edges (method calls) for this object
    all_edges = graph.edges.table()

    # Filter to this object's methods
    object_edges = []
    for i in range(len(all_edges)):
        if all_edges['object_name'][i] == object_name:
            object_edges.append({
                'method': all_edges['method_name'][i],
                'returns': all_edges['result_type'][i],
                'success': all_edges['success'][i]
            })

    if not object_edges:
        return f"No methods found in meta-graph for {object_name}"

    # Group by method name (remove duplicates)
    methods_map = {}
    for edge in object_edges:
        method = edge['method']
        if method not in methods_map:
            methods_map[method] = edge

    # Sort methods alphabetically
    sorted_methods = sorted(methods_map.items())

    # Generate markdown table
    lines = [
        "| Method | Returns | Status |",
        "|--------|---------|--------|"
    ]

    for method_name, info in sorted_methods:
        returns = info['returns'] if info['returns'] != 'unknown_return_type' else '?'
        status = "✓" if info['success'] else "✗"
        lines.append(f"| `{method_name}()` | `{returns}` | {status} |")

    return "\n".join(lines)


def generate_method_list_for_object(graph, object_name):
    """Generate a simple bullet list of methods."""

    all_edges = graph.edges.table()

    methods = set()
    for i in range(len(all_edges)):
        if all_edges['object_name'][i] == object_name:
            methods.add(all_edges['method_name'][i])

    if not methods:
        return f"No methods found for {object_name}"

    lines = []
    for method in sorted(methods):
        lines.append(f"- `{method}()`")

    return "\n".join(lines)


def main():
    print("Loading comprehensive test graph...")
    g = load_comprehensive_test_graph()

    core_objects = [
        'Graph',
        'Subgraph',
        'SubgraphArray',
        'NodesAccessor',
        'EdgesAccessor',
        'GraphTable',
        'NodesTable',
        'EdgesTable',
        'BaseTable',
        'BaseArray',
        'NumArray',
        'NodesArray',
        'EdgesArray',
        'GraphMatrix',
    ]

    print("\n" + "="*60)
    print("METHOD DOCUMENTATION FROM META-GRAPH")
    print("="*60 + "\n")

    for obj_name in core_objects:
        print(f"\n## {obj_name}\n")
        print(generate_method_table_for_object(g, obj_name))
        print()

    print("\n" + "="*60)
    print("To use in docs: Copy tables into 'Complete Method Reference' section")
    print("="*60)


if __name__ == "__main__":
    main()
