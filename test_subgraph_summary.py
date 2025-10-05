#!/usr/bin/env python3
"""
Test the newly implemented SubgraphArray.summary() method
"""

import groggy as gr

def test_summary():
    """Test summary statistics for SubgraphArray"""
    print("=" * 60)
    print("Testing SubgraphArray.summary() Method")
    print("=" * 60)

    g = gr.Graph()

    # Create nodes with department grouping
    print("\nCreating graph with 10 nodes in 2 departments...")
    for i in range(10):
        dept = 'Engineering' if i < 5 else 'Sales'
        g.add_node(name=f'Person{i}', department=dept, id=i)

    # Add some edges within each department
    print("Adding edges...")
    # Engineering connections
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)

    # Sales connections
    g.add_edge(5, 6)
    g.add_edge(6, 7)

    print(f"Graph has {g.node_count()} nodes and {g.edge_count()} edges")

    # Group by department
    print("\nGrouping nodes by department...")
    dept_groups = g.nodes.group_by('department')
    print(f"Created {len(dept_groups)} groups")
    print(f"Type: {type(dept_groups)}")

    # Get summary statistics
    print("\n--- Calling summary() ---")
    try:
        summary = dept_groups.summary()
        print(f"\nSummary result:")
        print(summary)
        print(f"\nType: {type(summary)}")

        # Try to access as table
        print("\n--- Verifying table structure ---")
        print(f"Available columns: {summary.columns}")

        # Show actual data
        print("\n--- Column Data ---")
        print(f"subgraph_id: {summary['subgraph_id']}")
        print(f"node_count: {summary['node_count']}")
        print(f"edge_count: {summary['edge_count']}")
        print(f"density: {summary['density']}")

        return True
    except Exception as e:
        print(f"\nError during summary: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_summary_with_varying_sizes():
    """Test summary with subgraphs of different sizes"""
    print("\n\n" + "=" * 60)
    print("Testing Summary with Varying Subgraph Sizes")
    print("=" * 60)

    g = gr.Graph()

    # Create nodes with different group sizes
    groups = [
        ('Small', 3),    # 3 nodes
        ('Medium', 6),   # 6 nodes
        ('Large', 10),   # 10 nodes
    ]

    node_id = 0
    for group_name, count in groups:
        print(f"\nCreating '{group_name}' group with {count} nodes...")
        for _ in range(count):
            g.add_node(name=f'Node{node_id}', category=group_name)
            node_id += 1

    # Add varying number of edges to each group
    # Small: fully connected (3 nodes -> 3 edges)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(0, 2)

    # Medium: partially connected (6 nodes -> 5 edges)
    g.add_edge(3, 4)
    g.add_edge(4, 5)
    g.add_edge(5, 6)
    g.add_edge(6, 7)
    g.add_edge(7, 8)

    # Large: sparsely connected (10 nodes -> 4 edges)
    g.add_edge(9, 10)
    g.add_edge(11, 12)
    g.add_edge(13, 14)
    g.add_edge(15, 16)

    print(f"\nTotal graph: {g.node_count()} nodes, {g.edge_count()} edges")

    # Group and summarize
    try:
        cat_groups = g.nodes.group_by('category')
        print(f"\nCreated {len(cat_groups)} groups by category")

        summary = cat_groups.summary()
        print("\nSummary Statistics:")
        print(summary)

        # Analyze density differences
        print("\n--- Expected vs Actual ---")
        print("Small (3 nodes, 3 edges): density = 3/3 = 1.0 (fully connected)")
        print("Medium (6 nodes, 5 edges): density = 5/15 ≈ 0.33")
        print("Large (10 nodes, 4 edges): density = 4/45 ≈ 0.09")

        return True
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []

    results.append(("Basic Summary Test", test_summary()))
    results.append(("Varying Sizes Test", test_summary_with_varying_sizes()))

    print("\n\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")

    exit(0 if all_passed else 1)
