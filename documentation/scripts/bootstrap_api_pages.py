#!/usr/bin/env python3
"""
Bootstrap API documentation pages from comprehensive test data.

This script:
1. Reads the comprehensive test graph
2. Extracts method information for each object
3. Creates API page skeletons with mkdocstrings blocks
4. Includes placeholders for hand-crafted sections
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python-groggy" / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import groggy as gr
from comprehensive_library_testing import load_comprehensive_test_graph


def get_methods_for_object(graph, object_name):
    """Extract all methods for a given object from the meta-graph."""
    methods = graph.edges[graph.edges['object_name'] == object_name]
    method_names = sorted(list(methods['method_name'].unique()))
    return method_names


def generate_mkdocstrings_block(object_name, methods):
    """Generate mkdocstrings YAML block for an object."""
    lines = [
        f"::: groggy.{object_name}",
        "    options:",
        "      show_root_heading: false",
        "      show_source: false",
        "      heading_level: 3",
        "      members:",
    ]

    for method in methods:
        lines.append(f"        - {method}")

    return "\n".join(lines)


def create_api_page_skeleton(object_name, methods, output_dir):
    """Create a skeleton API page for an object."""

    # Read template
    template_path = Path(__file__).parent.parent / "templates" / "api_page_template.md"
    with open(template_path, 'r') as f:
        template = f.read()

    # Generate mkdocstrings block
    mkdoc_block = generate_mkdocstrings_block(object_name, methods)

    # Basic replacements
    content = template.replace("{OBJECT_NAME}", object_name)
    content = content.replace("{Core component 1}", "TODO: Fill in")
    content = content.replace("{Core component 2}", "TODO: Fill in")
    content = content.replace("{Core component 3}", "TODO: Fill in")
    content = content.replace("        - {method1}\n        - {method2}\n        - {method3}\n        # ... add all methods here",
                            "\n".join([f"        - {m}" for m in methods]))

    # Write to file
    filename = object_name.lower().replace("_", "-") + ".md"
    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        f.write(content)

    return output_path


def main():
    # Load meta-graph
    print("Loading comprehensive test graph...")
    g = load_comprehensive_test_graph()

    # Core API objects to document
    core_objects = [
        'Graph',
        'Subgraph',
        'SubgraphArray',
        'GraphTable',
        'NodesTable',
        'EdgesTable',
        'BaseTable',
        'BaseArray',
        'NumArray',
        'NodesArray',
        'EdgesArray',
        'GraphMatrix',
        'NodesAccessor',
        'EdgesAccessor',
    ]

    # Output directory
    output_dir = Path(__file__).parent.parent.parent / "docs" / "api"
    output_dir.mkdir(exist_ok=True)

    # Generate pages
    print(f"\nGenerating API page skeletons in {output_dir}/\n")

    for obj_name in core_objects:
        # Get methods from meta-graph
        methods = get_methods_for_object(g, obj_name)

        # Skip if object exists but has no methods in our test data
        if len(methods) == 0:
            print(f"⚠️  {obj_name:20} - No methods found in test data, skipping")
            continue

        # Check if page already exists
        filename = obj_name.lower().replace("_", "-") + ".md"
        output_path = output_dir / filename

        if output_path.exists():
            print(f"✓  {obj_name:20} - Already exists ({len(methods)} methods)")
        else:
            # Create skeleton
            create_api_page_skeleton(obj_name, methods, output_dir)
            print(f"✓  {obj_name:20} - Created skeleton ({len(methods)} methods)")

    print("\n" + "="*60)
    print("Next steps:")
    print("1. Review generated skeletons in docs/api/")
    print("2. Fill in hand-crafted sections (Overview, Architecture, etc.)")
    print("3. Verify mkdocstrings blocks render correctly")
    print("4. Update mkdocs.yml navigation if needed")
    print("="*60)


if __name__ == "__main__":
    main()
