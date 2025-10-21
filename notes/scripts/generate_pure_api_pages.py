#!/usr/bin/env python3
"""
Generate pure API reference pages from comprehensive test meta-graph.

This script creates systematic API documentation for all core objects,
focusing only on method reference without mixing in theory or usage patterns.
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python-groggy" / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import groggy as gr
from comprehensive_library_testing import load_comprehensive_test_graph


def get_methods_for_object(graph, object_name):
    """Extract all methods for a given object from the meta-graph."""
    all_edges = graph.edges.table()

    methods = []
    for i in range(len(all_edges)):
        if all_edges['object_name'][i] == object_name:
            methods.append({
                'method': all_edges['method_name'][i],
                'returns': all_edges['result_type'][i],
                'success': all_edges['success'][i]
            })

    # Remove duplicates, keeping first occurrence
    seen = set()
    unique_methods = []
    for m in methods:
        if m['method'] not in seen:
            seen.add(m['method'])
            unique_methods.append(m)

    return sorted(unique_methods, key=lambda x: x['method'])


def categorize_methods(methods):
    """Categorize methods by their purpose."""
    categories = {
        'creation': [],
        'query': [],
        'transformation': [],
        'algorithm': [],
        'state': [],
        'io': []
    }

    for m in methods:
        method_name = m['method']

        # Creation & Construction
        if method_name in ['add_node', 'add_edge', 'add_nodes', 'add_edges',
                          'create', 'build', 'from_', 'load']:
            categories['creation'].append(m)

        # I/O & Export
        elif method_name in ['save', 'to_csv', 'to_parquet', 'to_pandas',
                            'to_networkx', 'to_json', 'save_bundle', 'load_bundle',
                            'to_graph', 'to_matrix', 'to_array']:
            categories['io'].append(m)

        # Algorithms
        elif method_name in ['connected_components', 'shortest_path', 'bfs', 'dfs',
                            'pagerank', 'clustering_coefficient', 'centrality',
                            'community_detection', 'spectral', 'neighborhood']:
            categories['algorithm'].append(m)

        # Transformations (delegations to other object types)
        elif method_name in ['table', 'nodes', 'edges', 'subgraph', 'sample',
                            'filter', 'select', 'group_by', 'agg', 'unique',
                            'to_nodes', 'to_edges']:
            categories['transformation'].append(m)

        # State Management
        elif method_name in ['set_attrs', 'set_node_attrs', 'set_edge_attrs',
                            'remove_node', 'remove_edge', 'clear', 'reset']:
            categories['state'].append(m)

        # Queries & Inspection (default for remaining)
        else:
            categories['query'].append(m)

    return categories


def generate_method_table(methods):
    """Generate markdown table of all methods."""
    if not methods:
        return "_No methods found in meta-graph testing._"

    lines = [
        "| Method | Returns | Status |",
        "|--------|---------|--------|"
    ]

    for m in methods:
        returns = m['returns'] if m['returns'] != 'unknown_return_type' else '?'
        status = "✓" if m['success'] else "✗"
        lines.append(f"| `{m['method']}()` | `{returns}` | {status} |")

    return "\n".join(lines)


def generate_category_section(category_methods):
    """Generate a simple list of methods in a category."""
    if not category_methods:
        return "_None in this category._\n"

    lines = []
    for m in category_methods:
        returns = m['returns'] if m['returns'] != 'unknown_return_type' else '?'
        lines.append(f"- **`{m['method']}()`** → `{returns}`")

    return "\n".join(lines) + "\n"


# Object metadata
OBJECT_METADATA = {
    'Graph': {
        'description': 'The core mutable graph object containing nodes, edges, and attributes.',
        'use_cases': [
            'Building and modifying graph structures',
            'Running graph algorithms',
            'Querying and filtering graph data'
        ],
        'related': ['Subgraph', 'GraphTable', 'NodesAccessor', 'EdgesAccessor'],
        'transformations': [
            '**Graph → Subgraph**: `g.nodes[condition]`, `g.subgraph(nodes=[...])`',
            '**Graph → GraphTable**: `g.table()`',
            '**Graph → NodesAccessor**: `g.nodes`',
            '**Graph → EdgesAccessor**: `g.edges`',
            '**Graph → ComponentsArray**: `g.connected_components()`'
        ],
        'guide_link': '../guide/graph-core.md'
    },
    'Subgraph': {
        'description': 'An immutable view into a subset of a Graph without copying data.',
        'use_cases': [
            'Filtering nodes/edges by conditions',
            'Working with portions of large graphs',
            'Creating temporary working sets without copying'
        ],
        'related': ['Graph', 'SubgraphArray', 'GraphTable'],
        'transformations': [
            '**Subgraph → Graph**: `sub.to_graph()`',
            '**Subgraph → GraphTable**: `sub.table()`',
            '**Subgraph → NodesAccessor**: `sub.nodes`',
            '**Subgraph → EdgesAccessor**: `sub.edges`',
            '**Subgraph → GraphMatrix**: `sub.to_matrix()`'
        ],
        'guide_link': '../guide/subgraphs.md'
    },
    'SubgraphArray': {
        'description': 'A collection of Subgraph objects, typically from algorithms like connected_components.',
        'use_cases': [
            'Working with graph components',
            'Analyzing community structures',
            'Processing multiple subgraphs in parallel'
        ],
        'related': ['Subgraph', 'Graph', 'GraphTable'],
        'transformations': [
            '**SubgraphArray → Subgraph**: `arr[0]`, `arr.sample(n)`',
            '**SubgraphArray → GraphTable**: `arr.table()`',
            '**SubgraphArray → SubgraphArray**: `arr.neighborhood(depth=2)`'
        ],
        'guide_link': '../guide/subgraphs.md'
    },
    'NodesAccessor': {
        'description': 'Accessor for node-level operations and filtering on graphs and subgraphs.',
        'use_cases': [
            'Filtering nodes by attributes',
            'Accessing node properties',
            'Creating node-based subgraphs'
        ],
        'related': ['Graph', 'Subgraph', 'NodesTable', 'NodesArray'],
        'transformations': [
            '**NodesAccessor → Subgraph**: `g.nodes[condition]`',
            '**NodesAccessor → NodesTable**: `g.nodes.table()`',
            '**NodesAccessor → NodesArray**: `g.nodes.ids()`',
            '**NodesAccessor → BaseArray**: `g.nodes["attribute"]`'
        ],
        'guide_link': '../guide/accessors.md'
    },
    'EdgesAccessor': {
        'description': 'Accessor for edge-level operations and filtering on graphs and subgraphs.',
        'use_cases': [
            'Filtering edges by attributes',
            'Accessing edge properties',
            'Creating edge-based subgraphs'
        ],
        'related': ['Graph', 'Subgraph', 'EdgesTable', 'EdgesArray'],
        'transformations': [
            '**EdgesAccessor → Subgraph**: `g.edges[condition]`',
            '**EdgesAccessor → EdgesTable**: `g.edges.table()`',
            '**EdgesAccessor → EdgesArray**: `g.edges.ids()`',
            '**EdgesAccessor → BaseArray**: `g.edges["attribute"]`'
        ],
        'guide_link': '../guide/accessors.md'
    },
    'GraphTable': {
        'description': 'Tabular representation of graph data with separate nodes and edges tables.',
        'use_cases': [
            'Exporting graph data to CSV/Parquet',
            'Converting to pandas DataFrames',
            'Tabular analysis of graph data'
        ],
        'related': ['Graph', 'Subgraph', 'NodesTable', 'EdgesTable'],
        'transformations': [
            '**GraphTable → NodesTable**: `table.nodes`',
            '**GraphTable → EdgesTable**: `table.edges`',
            '**GraphTable → DataFrame**: `table.to_pandas()`',
            '**GraphTable → Files**: `table.to_csv()`, `table.to_parquet()`'
        ],
        'guide_link': '../guide/tables.md'
    },
    'NodesTable': {
        'description': 'Tabular view of node data with columns for node attributes.',
        'use_cases': [
            'Analyzing node attributes in tabular form',
            'Aggregating node data',
            'Exporting node information'
        ],
        'related': ['GraphTable', 'BaseTable', 'NodesAccessor'],
        'transformations': [
            '**NodesTable → BaseArray**: `nodes_table["column"]`',
            '**NodesTable → DataFrame**: `nodes_table.to_pandas()`',
            '**NodesTable → AggregationResult**: `nodes_table.agg({"age": "mean"})`'
        ],
        'guide_link': '../guide/tables.md'
    },
    'EdgesTable': {
        'description': 'Tabular view of edge data with columns for edge attributes.',
        'use_cases': [
            'Analyzing edge attributes in tabular form',
            'Aggregating edge data',
            'Exporting edge information'
        ],
        'related': ['GraphTable', 'BaseTable', 'EdgesAccessor'],
        'transformations': [
            '**EdgesTable → BaseArray**: `edges_table["column"]`',
            '**EdgesTable → DataFrame**: `edges_table.to_pandas()`',
            '**EdgesTable → AggregationResult**: `edges_table.agg({"weight": "sum"})`'
        ],
        'guide_link': '../guide/tables.md'
    },
    'BaseTable': {
        'description': 'Base class for tabular data operations shared by NodesTable and EdgesTable.',
        'use_cases': [
            'Generic table operations',
            'Column-based data access',
            'Aggregations and transformations'
        ],
        'related': ['NodesTable', 'EdgesTable', 'BaseArray'],
        'transformations': [
            '**BaseTable → BaseArray**: `table["column"]`',
            '**BaseTable → DataFrame**: `table.to_pandas()`',
            '**BaseTable → AggregationResult**: `table.agg(...)`'
        ],
        'guide_link': '../guide/tables.md'
    },
    'BaseArray': {
        'description': 'Generic array for attribute columns with support for filtering and operations.',
        'use_cases': [
            'Working with graph attribute columns',
            'Filtering and masking',
            'Statistical operations on attributes'
        ],
        'related': ['NumArray', 'NodesArray', 'EdgesArray'],
        'transformations': [
            '**BaseArray → NumArray**: Automatic for numeric data',
            '**BaseArray → ndarray**: `array.to_numpy()`',
            '**BaseArray → list**: `list(array)`'
        ],
        'guide_link': '../guide/arrays.md'
    },
    'NumArray': {
        'description': 'Numeric array with mathematical operations and statistics.',
        'use_cases': [
            'Numerical computations on graph attributes',
            'Statistical analysis',
            'Vector operations'
        ],
        'related': ['BaseArray', 'NodesArray', 'EdgesArray'],
        'transformations': [
            '**NumArray → ndarray**: `num_array.to_numpy()`',
            '**NumArray → scalar**: `num_array.mean()`, `num_array.sum()`'
        ],
        'guide_link': '../guide/arrays.md'
    },
    'NodesArray': {
        'description': 'Array of node IDs with node-specific operations.',
        'use_cases': [
            'Working with collections of node IDs',
            'Node set operations',
            'Batch node queries'
        ],
        'related': ['NumArray', 'NodesAccessor'],
        'transformations': [
            '**NodesArray → Subgraph**: `g.nodes[node_array]`',
            '**NodesArray → ndarray**: `node_array.to_numpy()`'
        ],
        'guide_link': '../guide/arrays.md'
    },
    'EdgesArray': {
        'description': 'Array of edge IDs with edge-specific operations.',
        'use_cases': [
            'Working with collections of edge IDs',
            'Edge set operations',
            'Batch edge queries'
        ],
        'related': ['NumArray', 'EdgesAccessor'],
        'transformations': [
            '**EdgesArray → Subgraph**: `g.edges[edge_array]`',
            '**EdgesArray → ndarray**: `edge_array.to_numpy()`'
        ],
        'guide_link': '../guide/arrays.md'
    },
    'GraphMatrix': {
        'description': 'Matrix representation of graph data (adjacency, Laplacian, embeddings).',
        'use_cases': [
            'Matrix-based graph algorithms',
            'Spectral analysis',
            'Graph embeddings'
        ],
        'related': ['Graph', 'Subgraph'],
        'transformations': [
            '**GraphMatrix → ndarray**: `matrix.to_numpy()`',
            '**GraphMatrix → sparse**: `matrix.to_sparse()`'
        ],
        'guide_link': '../guide/matrices.md'
    }
}


def create_api_reference_page(object_name, methods, output_dir):
    """Create a pure API reference page for an object."""

    # Read template
    template_path = Path(__file__).parent.parent / "templates" / "pure_api_reference_template.md"
    with open(template_path, 'r') as f:
        template = f.read()

    # Get metadata
    metadata = OBJECT_METADATA.get(object_name, {
        'description': 'TODO: Add description',
        'use_cases': ['TODO: Add use case'],
        'related': [],
        'transformations': ['TODO: Add transformations'],
        'guide_link': '../guide/TODO.md'
    })

    # Generate method table
    method_table = generate_method_table(methods)

    # Categorize methods
    categories = categorize_methods(methods)

    # Build content
    content = template.replace("{OBJECT_NAME}", object_name)
    content = content.replace("{ONE_SENTENCE_DESCRIPTION}", metadata['description'])

    # Use cases
    use_cases_text = "\n".join(f"- {uc}" for uc in metadata['use_cases'])
    content = content.replace("- {USE_CASE_1}\n- {USE_CASE_2}\n- {USE_CASE_3}", use_cases_text)

    # Related objects
    related_text = "\n".join(f"- `{obj}`" for obj in metadata['related'])
    content = content.replace("- {RELATED_OBJECT_1}\n- {RELATED_OBJECT_2}", related_text)

    # Method table
    content = content.replace("{METHOD_TABLE}", method_table)

    # Category sections
    content = content.replace("{CREATION_METHODS}", generate_category_section(categories['creation']))
    content = content.replace("{QUERY_METHODS}", generate_category_section(categories['query']))
    content = content.replace("{TRANSFORMATION_METHODS}", generate_category_section(categories['transformation']))
    content = content.replace("{ALGORITHM_METHODS}", generate_category_section(categories['algorithm']))
    content = content.replace("{STATE_METHODS}", generate_category_section(categories['state']))
    content = content.replace("{IO_METHODS}", generate_category_section(categories['io']))

    # Transformations
    transformations_text = "\n".join(f"- {t}" for t in metadata['transformations'])
    content = content.replace("{TRANSFORMATION_LIST}", transformations_text)

    # Guide link
    content = content.replace("{GUIDE_LINK}", metadata['guide_link'])

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

    # Output directory
    output_dir = Path(__file__).parent.parent.parent / "docs" / "api"
    output_dir.mkdir(exist_ok=True)

    # Generate pages
    print(f"\nGenerating pure API reference pages in {output_dir}/\n")

    for obj_name in core_objects:
        # Get methods from meta-graph
        methods = get_methods_for_object(g, obj_name)

        # Skip if object exists but has no methods in our test data
        if len(methods) == 0:
            print(f"⚠️  {obj_name:20} - No methods found in test data, skipping")
            continue

        # Create API reference page
        output_path = create_api_reference_page(obj_name, methods, output_dir)
        print(f"✓  {obj_name:20} - Created API reference ({len(methods)} methods)")

    print("\n" + "="*60)
    print("Pure API reference pages generated successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review generated API reference pages in docs/api/")
    print("2. Create corresponding theory/usage guides in docs/guide/")
    print("3. Update mkdocs.yml navigation if needed")
    print("="*60)


if __name__ == "__main__":
    main()
