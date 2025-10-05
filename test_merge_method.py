#!/usr/bin/env python3
"""
Test the newly implemented merge() method for SubgraphArray
"""

import groggy as gr

def test_merge_basic():
    """Test basic merge functionality"""
    print("=" * 60)
    print("Testing SubgraphArray.merge() - Basic")
    print("=" * 60)

    # Create graph with departments
    g = gr.Graph()

    # Engineering department
    eng_nodes = []
    for i in range(5):
        nid = g.add_node(name=f'Eng{i}', department='Engineering', level=i+1)
        eng_nodes.append(nid)

    # Sales department
    sales_nodes = []
    for i in range(4):
        nid = g.add_node(name=f'Sales{i}', department='Sales', level=i+1)
        sales_nodes.append(nid)

    # Add edges within Engineering
    g.add_edge(eng_nodes[0], eng_nodes[1], weight=10)
    g.add_edge(eng_nodes[1], eng_nodes[2], weight=15)
    g.add_edge(eng_nodes[2], eng_nodes[3], weight=20)

    # Add edges within Sales
    g.add_edge(sales_nodes[0], sales_nodes[1], weight=5)
    g.add_edge(sales_nodes[1], sales_nodes[2], weight=8)

    print(f"\nOriginal graph: {g.node_count()} nodes, {g.edge_count()} edges")

    try:
        # Group by department
        dept_groups = g.nodes.group_by('department')
        print(f"Created {len(dept_groups)} department groups")

        # Merge back
        print("\n--- Merging subgraphs ---")
        merged = dept_groups.merge()
        print(f"Merged graph type: {type(merged)}")
        print(f"Merged graph: {merged.node_count()} nodes, {merged.edge_count()} edges")

        # Verify counts
        assert merged.node_count() == g.node_count(), "Node count mismatch"
        assert merged.edge_count() == g.edge_count(), "Edge count mismatch"

        # Verify attributes preserved
        print("\n--- Verifying attributes ---")
        for nid in range(merged.node_count()):
            orig_name = g.get_node_attr(nid, 'name')
            merged_name = merged.get_node_attr(nid, 'name')
            assert orig_name == merged_name, f"Name mismatch for node {nid}"

        print("✓ All attributes preserved correctly")

        return True
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_merge_with_disconnected_components():
    """Test merge with disconnected components"""
    print("\n\n" + "=" * 60)
    print("Testing merge() with Disconnected Components")
    print("=" * 60)

    g = gr.Graph()

    # Create 3 disconnected components
    comp1 = [g.add_node(component=1, value=i) for i in range(3)]
    comp2 = [g.add_node(component=2, value=i+10) for i in range(3)]
    comp3 = [g.add_node(component=3, value=i+20) for i in range(3)]

    # Add edges within each component
    g.add_edge(comp1[0], comp1[1])
    g.add_edge(comp1[1], comp1[2])

    g.add_edge(comp2[0], comp2[1])
    g.add_edge(comp2[1], comp2[2])

    g.add_edge(comp3[0], comp3[1])

    print(f"\nOriginal graph: {g.node_count()} nodes, {g.edge_count()} edges")

    try:
        # Group by component
        comp_groups = g.nodes.group_by('component')
        print(f"Created {len(comp_groups)} component groups")

        # Get summary before merge
        summary = comp_groups.summary()
        print(f"\nComponent summary before merge:")
        print(summary)

        # Merge
        merged = comp_groups.merge()
        print(f"\nMerged graph: {merged.node_count()} nodes, {merged.edge_count()} edges")

        # Verify all nodes and edges preserved
        assert merged.node_count() == 9, f"Expected 9 nodes, got {merged.node_count()}"
        assert merged.edge_count() == 5, f"Expected 5 edges, got {merged.edge_count()}"

        print("✓ Disconnected components merged correctly")

        return True
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_merge_preserves_edge_attributes():
    """Test that merge preserves edge attributes"""
    print("\n\n" + "=" * 60)
    print("Testing merge() Preserves Edge Attributes")
    print("=" * 60)

    g = gr.Graph()

    # Create nodes
    nodes = []
    for i in range(6):
        cat = 'A' if i < 3 else 'B'
        nid = g.add_node(name=f'N{i}', category=cat)
        nodes.append(nid)

    # Add edges with attributes
    edges = [
        (0, 1, 'friendship', 10),
        (1, 2, 'work', 5),
        (3, 4, 'friendship', 8),
        (4, 5, 'family', 15),
    ]

    for src, tgt, edge_type, strength in edges:
        g.add_edge(nodes[src], nodes[tgt], type=edge_type, strength=strength)

    print(f"\nOriginal graph: {g.node_count()} nodes, {g.edge_count()} edges")

    try:
        # Group and merge
        cat_groups = g.nodes.group_by('category')
        merged = cat_groups.merge()

        print(f"Merged graph: {merged.node_count()} nodes, {merged.edge_count()} edges")

        # Check that merged graph is functional
        print("\n--- Testing merged graph functionality ---")
        table = merged.nodes.table()
        print(f"Nodes table: {len(table)} rows")

        edges_table = merged.edges.table()
        print(f"Edges table: {len(edges_table)} rows")

        print("✓ Merged graph is fully functional")

        return True
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []

    results.append(("Basic merge", test_merge_basic()))
    results.append(("Merge with disconnected components", test_merge_with_disconnected_components()))
    results.append(("Merge preserves edge attributes", test_merge_preserves_edge_attributes()))

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
