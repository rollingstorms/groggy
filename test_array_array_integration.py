#!/usr/bin/env python3
"""
Integration test for ArrayArray with TableArray and SubgraphArray
Tests the new pandas-like groupby API: g.nodes.group_by('col')['val'].mean()
"""

import groggy as gr

def test_node_groupby_with_extraction():
    """Test node groupby with attribute extraction"""
    print("\n=== Testing Node GroupBy with Attribute Extraction ===")

    # Create a graph with node attributes
    g = gr.Graph()

    # Add nodes with attributes for grouping
    n1 = g.add_node(name="Alice", department="Engineering", salary=100)
    n2 = g.add_node(name="Bob", department="Engineering", salary=120)
    n3 = g.add_node(name="Carol", department="Sales", salary=90)
    n4 = g.add_node(name="Dave", department="Sales", salary=110)
    n5 = g.add_node(name="Eve", department="Engineering", salary=130)

    # Add some edges
    g.add_edge(n1, n2)
    g.add_edge(n2, n3)
    g.add_edge(n3, n4)
    g.add_edge(n1, n5)

    print(f"Created graph with {g.node_count()} nodes and {g.edge_count()} edges")

    try:
        # Group nodes by department
        dept_groups = g.nodes.group_by("department")
        print(f"\nGrouped nodes by 'department': {len(dept_groups)} groups")
        print(f"Type: {type(dept_groups)}")

        # Extract salary attribute from all groups
        salary_arrays = dept_groups["salary"]
        print(f"\nExtracted 'salary' attribute: {salary_arrays}")
        print(f"Type: {type(salary_arrays)}")

        # Calculate mean salary per department
        mean_salaries = salary_arrays.mean()
        print(f"\nMean salaries per department: {mean_salaries}")
        print(f"Type: {type(mean_salaries)}")

        return True
    except Exception as e:
        print(f"\nError during node groupby: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_subgraph_array_attribute_extraction():
    """Test SubgraphArray attribute extraction with __getitem__"""
    print("\n\n=== Testing SubgraphArray Attribute Extraction ===")

    # Create a graph with node attributes
    g = gr.Graph()

    # Add nodes with attributes
    n1 = g.add_node(name="Alice", department="Engineering", level=3)
    n2 = g.add_node(name="Bob", department="Engineering", level=2)
    n3 = g.add_node(name="Carol", department="Sales", level=4)
    n4 = g.add_node(name="Dave", department="Sales", level=2)
    n5 = g.add_node(name="Eve", department="Engineering", level=5)

    # Add some edges
    g.add_edge(n1, n2)
    g.add_edge(n2, n3)
    g.add_edge(n3, n4)
    g.add_edge(n1, n5)

    print(f"Created graph with {g.node_count()} nodes and {g.edge_count()} edges")

    # Group nodes by department (returns SubgraphArray)
    dept_groups = g.nodes.group_by("department")
    print(f"\nGrouped nodes by 'department': {len(dept_groups)} groups")

    # Extract level attribute from all groups
    try:
        level_arrays = dept_groups["level"]
        print(f"\nExtracted 'level' attribute: {level_arrays}")
        print(f"Type: {type(level_arrays)}")

        # Calculate mean level per department
        mean_levels = level_arrays.mean()
        print(f"\nMean levels per department: {mean_levels}")
        print(f"Type: {type(mean_levels)}")

        return True
    except Exception as e:
        print(f"\nError during attribute extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregation_methods():
    """Test all ArrayArray aggregation methods"""
    print("\n\n=== Testing ArrayArray Aggregation Methods ===")

    # Create a graph with node attributes
    g = gr.Graph()

    # Add nodes with numeric attributes
    for i in range(12):
        dept = "Eng" if i < 6 else "Sales"
        score = float(i + 1)  # 1-12
        g.add_node(department=dept, score=score, bonus=score * 2)

    print(f"Created graph with {g.node_count()} nodes")

    try:
        # Group by department
        dept_groups = g.nodes.group_by("department")
        print(f"\nGrouped nodes by 'department': {len(dept_groups)} groups")

        # Extract score attribute
        score_arrays = dept_groups["score"]
        print(f"Score arrays: {score_arrays}")

        # Test all aggregation methods
        print("\n--- Testing Aggregation Methods ---")

        mean_result = score_arrays.mean()
        print(f"mean(): {mean_result}")

        sum_result = score_arrays.sum()
        print(f"sum(): {sum_result}")

        min_result = score_arrays.min()
        print(f"min(): {min_result}")

        max_result = score_arrays.max()
        print(f"max(): {max_result}")

        std_result = score_arrays.std()
        print(f"std(): {std_result}")

        count_result = score_arrays.count()
        print(f"count(): {count_result}")

        return True
    except Exception as e:
        print(f"\nError during aggregation test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("ArrayArray Integration Test Suite")
    print("=" * 60)

    results = []

    results.append(("Node GroupBy with Extraction", test_node_groupby_with_extraction()))
    results.append(("SubgraphArray Attribute Extraction", test_subgraph_array_attribute_extraction()))
    results.append(("ArrayArray Aggregation Methods", test_aggregation_methods()))

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
