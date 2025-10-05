#!/usr/bin/env python3
"""
Test the newly implemented map() methods for SubgraphArray and TableArray
"""

import groggy as gr

def test_subgraph_array_map():
    """Test SubgraphArray.map() method"""
    print("=" * 60)
    print("Testing SubgraphArray.map() Method")
    print("=" * 60)

    g = gr.Graph()

    # Create nodes with department grouping
    print("\nCreating graph with nodes in 3 departments...")
    for i in range(15):
        if i < 5:
            dept = 'Engineering'
        elif i < 10:
            dept = 'Sales'
        else:
            dept = 'Marketing'
        g.add_node(name=f'Person{i}', department=dept, score=float(i + 1))

    # Add some edges
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)  # Engineering: 4 edges

    g.add_edge(5, 6)
    g.add_edge(6, 7)
    g.add_edge(7, 8)  # Sales: 3 edges

    g.add_edge(10, 11)
    g.add_edge(11, 12)  # Marketing: 2 edges

    print(f"Total graph: {g.node_count()} nodes, {g.edge_count()} edges")

    try:
        # Group by department
        dept_groups = g.nodes.group_by('department')
        print(f"\nCreated {len(dept_groups)} department groups")

        # Test 1: Map node_count over subgraphs
        print("\n--- Test 1: Map node_count() ---")
        node_counts = dept_groups.map(lambda sg: sg.node_count())
        print(f"Node counts: {node_counts}")
        print(f"Type: {type(node_counts)}")

        # Test 2: Map edge_count over subgraphs
        print("\n--- Test 2: Map edge_count() ---")
        edge_counts = dept_groups.map(lambda sg: sg.edge_count())
        print(f"Edge counts: {edge_counts}")

        # Test 3: Map with float calculation (density)
        print("\n--- Test 3: Map density calculation ---")
        densities = dept_groups.map(lambda sg:
            sg.edge_count() / ((sg.node_count() * (sg.node_count() - 1)) / 2.0)
            if sg.node_count() > 1 else 0.0
        )
        print(f"Densities: {densities}")

        return True
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_table_array_map():
    """Test TableArray.map() method (via nodes_table)"""
    print("\n\n" + "=" * 60)
    print("Testing TableArray.map() Method")
    print("=" * 60)

    g = gr.Graph()

    # Create nodes with grouping
    print("\nCreating graph with nodes in 2 categories...")
    for i in range(10):
        cat = 'A' if i < 5 else 'B'
        g.add_node(name=f'Node{i}', category=cat, value=float(i + 1))

    print(f"Total graph: {g.node_count()} nodes")

    try:
        # Group by category
        cat_groups = g.nodes.group_by('category')
        print(f"\nCreated {len(cat_groups)} category groups")

        # Get nodes tables for each subgraph
        print("\n--- Getting nodes_table() ---")
        tables = cat_groups.nodes_table()
        print(f"Tables: {tables}")
        print(f"Type: {type(tables)}")

        # Test: Map len() over tables
        print("\n--- Test: Map len() over tables ---")
        table_lengths = tables.map(lambda t: len(t))
        print(f"Table lengths: {table_lengths}")
        print(f"Type: {type(table_lengths)}")

        return True
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_map_with_strings():
    """Test map() with string return values"""
    print("\n\n" + "=" * 60)
    print("Testing map() with String Return Values")
    print("=" * 60)

    g = gr.Graph()

    # Create simple graph
    for i in range(6):
        dept = 'Eng' if i < 3 else 'Sales'
        g.add_node(name=f'P{i}', department=dept)

    try:
        dept_groups = g.nodes.group_by('department')

        # Map to string labels
        print("\n--- Mapping to string labels ---")
        labels = dept_groups.map(lambda sg:
            f"{sg.node_count()}_nodes"
        )
        print(f"Labels: {labels}")

        return True
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []

    results.append(("SubgraphArray.map()", test_subgraph_array_map()))
    results.append(("TableArray.map()", test_table_array_map()))
    results.append(("map() with strings", test_map_with_strings()))

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
